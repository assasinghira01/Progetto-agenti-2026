import os
import json
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from typing import Optional


from langchain_openai import ChatOpenAI


# . Schema Pydantic per Ricette, Sagre e Trend
class ContenutoWebEstratto(BaseModel):
    titolo: str = Field(description="Il titolo della pagina o dell'articolo.")
    tipo_contenuto: str = Field(
        description="Scrivi 'Ricetta' se contiene dosi, altrimenti 'Informativo'."
    )

    ingredienti: Optional[dict[str, str]] = Field(
        default=None,
        description="Dizionario degli ingredienti (SOLO se è una ricetta).",
    )
    procedimento_riassunto: Optional[list[str]] = Field(
        default=None, description="Passaggi della preparazione (SOLO se è una ricetta)."
    )
    dettagli_informativi: Optional[str] = Field(
        default=None,
        description="Luogo, date dell'evento, o riassunto dettagliato del trend (SOLO se NON è una ricetta).",
    )


llm_estrazione = ChatOpenAI(model="gpt-4o-mini", temperature=0)
estrattore_web = llm_estrazione.with_structured_output(ContenutoWebEstratto)


@tool
def esegui_ricerca_web(query: str) -> str:
    """
    Cerca su Internet informazioni legati al topic.
    Entra nelle pagine web ed estrae dati strutturati.
    """
    tool_tavily = TavilySearchResults(max_results=1, language="it")

    try:
        risultati = tool_tavily.invoke({"query": query})
        output_finale = ""

        for i, res in enumerate(risultati):
            url = res.get("url")

            try:
                # Scraping della pagina
                loader = WebBaseLoader(url)
                documenti = loader.load()
                testo_pagina = documenti[0].page_content
                testo_pulito = " ".join(testo_pagina.split())[:10000]

                # ESTRAZIONE STRUTTURATA (Usando il mini-LLM dedicato!)
                prompt_estrazione = f"Leggi questo testo web grezzo ed estrai le informazioni:\n\n{testo_pulito}"
                dati_strutturati = estrattore_web.invoke(
                    [HumanMessage(content=prompt_estrazione)]
                )

                output_finale += f"\n--- Fonte Web: {url} ---\n"
                # Usiamo exclude_none=True per non stampare campi vuoti (es. ingredienti nulli in una sagra)
                output_finale += (
                    json.dumps(
                        dati_strutturati.model_dump(exclude_none=True),
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                print(f"\n[Web Tool] Dati estratti con successo da {url}:")
                print(
                    json.dumps(
                        dati_strutturati.model_dump(exclude_none=True),
                        indent=2,
                        ensure_ascii=False,
                    )
                )

            except Exception as e_scrape:
                print(
                    f" [Web Tool] Scraping o estrazione fallita per {url}. Uso il riassunto di base."
                )
                output_finale += (
                    f"\n--- Fonte Web: {url} ---\nRiassunto: {res.get('content')}\n"
                )

        return output_finale

    except Exception as e:
        return f"Errore critico durante la ricerca web: {str(e)}"
