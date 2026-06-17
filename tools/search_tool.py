import os
import json
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


@tool
def esegui_ricerca_web(query: str) -> str:
    """Cerca ricette su Internet, le valuta per qualità e restituisce solo le migliori."""

    tool_tavily = TavilySearchResults(max_results=5, language="IT")
    risultati_grezzi = tool_tavily.invoke({"query": query})

    ricette_valutate = []

    for res in risultati_grezzi:
        url = res.get("url")
        try:
            # 1. Scraping
            loader = WebBaseLoader(url)
            testo = loader.load()[0].page_content
            testo_pulito = " ".join(testo.split())[:15000]

            # 2. Estrazione + valutazione qualità in un solo passaggio LLM
            dati = estrattore_web.invoke(
                [
                    HumanMessage(
                        content=f"Estrai e valuta questa ricetta:\n\n{testo_pulito}"
                    )
                ]
            )

            # 3. Scarta subito se non è una ricetta
            if dati.tipo_contenuto != "Ricetta":
                continue

            # 4. Score combinato: dominio + contenuto
            score_dominio = score_fonte(url)  # whitelist domini
            score_totale = score_dominio + dati.score_contenuto  # 0-10

            print(f"[Web Tool] {url}")
            print(f"  Score dominio: {score_dominio}/5")
            print(f"  Score contenuto: {dati.score_contenuto}/5")
            print(f"  Score totale: {score_totale}/10")

            ricette_valutate.append((score_totale, url, dati))

        except Exception as e:
            print(f"[Web Tool] Scraping fallito per {url}: {e}")
            continue

    # 5. Ordina per score e prendi le 2 migliori
    ricette_valutate.sort(key=lambda x: x[0], reverse=True)
    migliori = ricette_valutate[:2]
    print(f"{migliori}")
    # 6. Costruisci output solo con le migliori
    output = ""
    for score, url, dati in migliori:
        output += f"\n--- Fonte Web (score: {score}/10): {url} ---\n"
        output += json.dumps(
            dati.model_dump(exclude_none=True), indent=2, ensure_ascii=False
        )
        output += "\n"
        print(f"output")

    return output if output else "Nessuna ricetta di qualità sufficiente trovata."
