import os
import json
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
    "ricette.giallozafferano.it": 5,
    "blog.giallozafferano.it": 5,
    "cucchiaio.it": 5,
    "lacucinaitaliana.it": 5,
    "academiabarilla.com": 5,
    "tavolartegusto.it": 4,
    "ricette.it": 4,
    "misya.info": 3,
}

co = cohere.Client(os.getenv("COHERE_API_KEY"))


def score_fonte(url: str) -> int:
    from urllib.parse import urlparse

    dominio = urlparse(url).netloc.replace("www.", "")
    return DOMINI_AFFIDABILI.get(dominio, 2)


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

    # Qualità
    ha_dosi_precise: bool
    ha_procedimento_completo: bool
    ha_tempi_cottura: bool
    ha_numero_persone: bool
    sembra_tradizionale: bool

    @property
    def score_contenuto(self) -> int:
        return sum(
            [
                self.ha_dosi_precise,
                self.ha_procedimento_completo,
                self.ha_tempi_cottura,
                self.ha_numero_persone,
                self.sembra_tradizionale,
            ]
        )


llm_estrazione = ChatOpenAI(model="gpt-4o-mini", temperature=0)
estrattore_web = llm_estrazione.with_structured_output(ContenutoWebEstratto)


def ottimizza_query_ricerca(query_originale: str) -> str:
    prompt = f"""
    Analizza questa query di ricerca culinaria: '{query_originale}'
    Il tuo compito è trasformarla in una query ottimizzata per motori di ricerca, 
    focalizzata ESCLUSIVAMENTE sulla preparazione base o sull'ingrediente specifico, 
    rimuovendo il piatto finale in cui verrà usato.

    Esempi:
    - 'Ricetta ragù classico per lasagne' -> 'Ricetta ragù di carne classico bolognese'
    - 'Besciamella liquida per cannelloni' -> 'Ricetta besciamella dosi procedimento'
    - 'Pasta frolla friabile per crostata di marmellata' -> 'Ricetta pasta frolla friabile'
    
    Rispondi SOLO con la query ottimizzata, senza commenti.
    """
    return llm_estrazione.invoke(prompt).content


@tool
def esegui_ricerca_web(query: str) -> str:
    """Cerca, riordina con AI e infine estrae solo le migliori ricette."""

    # 1. Query Optimization
    query_ottimizzata = ottimizza_query_ricerca(query)

    # 2. Retrieval iniziale (più ampio: 10 risultati)
    tool_tavily = TavilySearchResults(max_results=10, language="IT")
    risultati_grezzi = tool_tavily.invoke({"query": query_ottimizzata})

    # 3. Re-Ranking con Cohere
    # Prepariamo i documenti per il reranker usando lo snippet di testo fornito da Tavily
    doc_snippets = [res.get("content", "") for res in risultati_grezzi]

    print(f"[Web Tool] Re-ranking di {len(risultati_grezzi)} risultati...")
    rerank_results = co.rerank(
        query=query_ottimizzata,
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

            score_totale = score_fonte(url) + dati.score_contenuto
            ricette_valutate.append((score_totale, url, dati))

        except Exception as e:
            print(f"[Web Tool] Scraping fallito per {url}: {e}")
            continue

    # 5. Ordina per score e prendi le 2 migliori
    ricette_valutate.sort(key=lambda x: x[0], reverse=True)
    migliori = ricette_valutate[:5]
    print(f"{migliori}")
    # 6. Costruisci output solo con le migliori
    output_list = []
    SEPARATORE = "\n\n|||SPLIT_DOC|||\n\n"
    for score, url, dati in migliori:
        testo_singolo = (
            f"=== FONTE WEB (score: {score}/10): {url} ===\n"
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
