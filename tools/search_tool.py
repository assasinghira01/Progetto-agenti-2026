import os
from langchain_community.tools.tavily_search import TavilySearchResults

def esegui_ricerca_web(query: str) -> str:
    """Cerca su Internet informazioni, varianti e trend culinari legati al topic."""
    tool_tavily = TavilySearchResults(max_results=2)
    
    try:
        risultati = tool_tavily.invoke({"query": query})
        testo_estratto = ""
        for i, res in enumerate(risultati):
            testo_estratto += f"\n--- Fonte Web {i+1}: {res.get('url')} ---\n{res.get('content')}\n"
        return testo_estratto
    except Exception as e:
        return f"Errore durante la ricerca web: {str(e)}"