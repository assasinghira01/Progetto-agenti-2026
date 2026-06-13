from langchain_core.tools import tool


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool per la riflessione strategica e decisionale dell'agente.

    Usa questo tool obbligatoriamente dopo ogni ricerca o controllo nel database
    per elaborare i dati trovati e dichiarare le tue intenzioni al sistema.

    REGOLE DI FORMATTAZIONE:
    La tua riflessione passata come parametro DEVE OBBLIGATORIAMENTE terminare
    con una di queste due esatte stringhe:
    - "STATO: CONTINUO" -> Se devi ancora cercare, verificare altri topic, o se un topic è stato BLOCCATO.
    - "STATO: FINITO" -> SOLO ed ESCLUSIVAMENTE quando hai trovato 3 topic diversi che hanno ricevuto tutti risposta "OK" dal controllo storico.

    Args:
        reflection: Il testo della tua riflessione contenente i ragionamenti e lo stato finale.

    Returns:
        Conferma che la riflessione è stata registrata nel sistema di routing.
    """
    print("[think_tool] Riflessione ricevuta")
    return f"Riflessione registrata con successo: {reflection}"
