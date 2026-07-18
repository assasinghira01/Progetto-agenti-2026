import os
import cohere
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from typing import Optional
from langchain_openai import ChatOpenAI
from graph.schemas import Ingrediente

DOMINI_AFFIDABILI = {
    "giallozafferano.it": 5,
    "cucchiaio.it": 5,
    "lacucinaitaliana.it": 5,
    "academiabarilla.com": 5,
    "tavolartegusto.it": 4,
    "ricette.it": 4,
    "misya.info": 3,
}
DOMINI_SPAZZATURA = [
    "wikipedia.org",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "instagram.com",
    "pinterest.it",
    "tripadvisor.it",
]
co = cohere.Client(os.getenv("COHERE_API_KEY"))


def score_fonte(url: str) -> int:
    from urllib.parse import urlparse

    dominio_estratto = urlparse(url).netloc.lower()
    dominio_estratto = dominio_estratto.replace("www.", "")
    for dominio_base, score in DOMINI_AFFIDABILI.items():
        if dominio_estratto == dominio_base or dominio_estratto.endswith(
            "." + dominio_base
        ):
            return score
    return 0


class ContenutoWebEstratto(BaseModel):
    titolo: str
    porzioni: Optional[str] = Field(
        default=None,
        description="Es: 'per 4 persone', 'per una teglia 20x30 cm', '6 porzioni'",
    )
    tipo_contenuto: str

    ingredienti: Optional[list[Ingrediente]] = Field(
        default=None,
        description="Lista ingredienti PRINCIPALI. Ingredienti accessori (per friggere, per ungere) vanno solo nel procedimento",
    )
    procedimento: Optional[list[str]] = Field(
        default=None,
        description="Lista di passaggi numerati, chiari e concisi. Massimo 8 passaggi. Gli ingredienti accessori vanno menzionati qui",
    )

    score_qualita: int = Field(
        ge=1,
        le=5,
        description="""
        Valuta la qualità complessiva della ricetta da 1 a 5 considerando:
        - completezza
        - chiarezza
        - precisione delle dosi
        - qualità del procedimento
        - aderenza alla ricetta tradizionale
        """,
    )


llm_estrazione = ChatOpenAI(model="gpt-4o-mini", temperature=0)
estrattore_web = llm_estrazione.with_structured_output(ContenutoWebEstratto)


def ottimizza_query_ricerca(query_originale: str) -> str:
    prompt = f"""
    Sei un esperto di search engine optimization per ricette culinarie.
    Il tuo compito è trasformare una query di ricerca in una versione ottimizzata per motori di ricerca, in lingua italiana.

    **Query originale:** "{query_originale}"

    **REGOLE RIGOROSE (applica nell'ordine):**

    1. **PIATTO BASE vs PREPARAZIONE**
       - Se la query descrive una preparazione per un piatto finale (es. "ragù per lasagne", "besciamella per cannelloni"), isola la preparazione e rimuovi il piatto finale. Aggiungi "ricetta" all'inizio.
         Esempi:
         - "ragù per lasagne" → "ricetta ragù di carne classico"
         - "besciamella per cannelloni" → "ricetta besciamella classica"
       - Se invece è un piatto autonomo (es. "ganache al cioccolato", "lasagne al pesto", "pizza capricciosa"), mantieni il nome completo senza rimuovere parti.

    2. **RILEVAZIONE E ESPANSIONE DELLA VARIANTE (FONDAMENTALE)**
       - Se la query originale contiene una VARIANTE ESPLICITA (es. "light", "senza uova", "vegana", "senza glutine", "senza lattosio", "al forno", "fritta", "allo zafferano"), devi **rafforzarla** aggiungendo le seguenti parole chiave per rendere la query più specifica e rilevante:
       - Esempi di espansione:
         - "carbonara light" → "carbonara ipocalorica ricetta leggera"
         - "besciamella light" → "besciamella ipocolariica leggera light"
         - "tiramisù senza glutine" → "tiramisù senza glutine per celiaci"
         - "tiramisu alle fragole->" → "tiramisu alle fragole"
         - "maionese vegana" → "maionese vegana senza uova"
         - "caronara vegana" → "caronara vegana senza uova e guanciale"
       - Se la variante NON è presente, NON aggiungere nulla.

    3. **AGGIUNTA FINALE**
       - Termina SEMPRE la query con le parole: "ricetta completa in italiano"

    **FORMATO DI RISPOSTA:**
    Restituisci ESCLUSIVAMENTE la query ottimizzata, senza virgolette, senza spiegazioni, senza commenti. Solo la stringa.
    """
    return llm_estrazione.invoke(prompt).content


@tool
def esegui_ricerca_web(query: str) -> str:
    """Cerca, riordina con AI e infine estrae solo le migliori ricette."""
    # query_ottimizzata = ottimizza_query_ricerca(query)
    # 2. Retrieval iniziale (più ampio: 10 risultati)
    print(f"\n[Web Tool] Query originale: '{query}'")
    # print(f"[Web Tool] Query ottimizzata: '{query_ottimizzata}'\n")
    tool_tavily = TavilySearchResults(
        max_results=10,
        exclude_domains=DOMINI_SPAZZATURA,
        country="it",
    )
    risultati_grezzi = tool_tavily.invoke({"query": query})

    # 3. Re-Ranking con Cohere
    # Prepariamo i documenti per il reranker usando lo snippet di testo fornito da Tavily
    doc_snippets = [res.get("content", "") for res in risultati_grezzi]

    print(f"[Web Tool] Re-ranking di {len(risultati_grezzi)} risultati...")
    rerank_results = co.rerank(
        query=query,
        documents=doc_snippets,
        model="rerank-multilingual-v3.0",
        top_n=5,  # Estraiamo solo i 3 migliori
    )

    # 4. Scraping e Estrazione SOLO sui Top 3
    ricette_valutate = []
    for rank in rerank_results.results:
        res = risultati_grezzi[rank.index]

        url = res.get("url")

        try:
            loader = WebBaseLoader(url)
            testo = loader.load()[0].page_content
            testo_pulito = " ".join(testo.split())[:15000]

            dati = estrattore_web.invoke(
                [
                    HumanMessage(
                        content=f"Valuta se è una ricetta singola:\n{testo_pulito}"
                    )
                ]
            )

            score_finale = (
                (rank.relevance_score * 5) + score_fonte(url) + dati.score_qualita
            )
            ricette_valutate.append((score_finale, url, dati))

        except Exception as e:
            print(f"[Web Tool] Scraping fallito per {url}: {e}")
            continue

    # 5. Ordina per score e prendi le 2 migliori
    print(f"{ricette_valutate}\n")
    print("QUELLE ORDINATE.")
    ricette_valutate.sort(key=lambda x: x[0], reverse=True)
    migliori = ricette_valutate[:2]
    print(f"{migliori}")
    # 6. Costruisci output solo con le migliori
    output_list = []
    SEPARATORE = "\n\n|||SPLIT_DOC|||\n\n"
    for score, url, dati in migliori:
        testo_singolo = (
            f"=== FONTE WEB (score: {score}/15): {url} ===\n"
            f"{formatta_ricetta_markdown(dati)}\n"
            f"==================="
        )
        output_list.append(testo_singolo)

    if output_list:
        # Uniamo con il separatore magico, non con una semplice stringa!
        return SEPARATORE.join(output_list)
    else:
        return "Nessuna ricetta di qualità sufficiente trovata."


def formatta_ricetta_markdown(dati) -> str:
    """
    Converte un oggetto Pydantic (ricetta estratta) in una stringa Markdown pulita.
    """
    output = ""

    # 1. Titolo
    titolo = getattr(dati, "titolo", "Titolo Sconosciuto")
    output += f"# {titolo}\n\n"

    # 2. Descrizione
    descrizione = getattr(dati, "descrizione", "")
    if descrizione:
        output += f"{descrizione}\n\n"

    # 3. Ingredienti
    output += "## Ingredienti\n"
    if hasattr(dati, "ingredienti") and dati.ingredienti:
        for ing in dati.ingredienti:
            output += f"- {str(ing)}\n"
    else:
        output += "- Nessun ingrediente specificato.\n"

    # 4. Procedimento
    output += "\n## Procedimento\n"
    if hasattr(dati, "procedimento") and dati.procedimento:
        for i, step in enumerate(dati.procedimento, 1):
            output += f"{i}. {step}\n"
    else:
        output += "1. Nessun procedimento specificato.\n"

    return output
