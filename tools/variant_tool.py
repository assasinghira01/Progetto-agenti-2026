from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def chiedi_variante(varianti: str) -> str:
    """
    Chiedi all'utente di scegliere una variante tra quelle proposte.
    Usa questo strumento SOLO dopo aver elencato le opzioni disponibili.

    Args:
        varianti: La lista delle varianti proposta (es. "1. Caponata al forno\n2. Caponata con cioccolato\n3. Caponata light")
    """
    print("\n[TOOL] Attendo scelta dell'utente...")
    scelta = interrupt(f"Scegli una variante:\n{varianti}")
    print(f"[TOOL] Utente ha scelto: {scelta}")
    return f"SCELTA_UTENTE|{scelta}"
