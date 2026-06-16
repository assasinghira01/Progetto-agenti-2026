from langchain_core.tools import tool


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool FONDAMENTALE per la riflessione strategica e decisionale dell'agente.

    Usa questo tool obbligatoriamente dopo aver raccolto informazioni per spiegare
    il tuo ragionamento logico e dichiarare la tua prossima mossa al sistema.

    REGOLE DI FORMATTAZIONE OBBLIGATORIE:
    Il parametro 'reflection' DEVE terminare SEMPRE con una di queste due stringhe:
    "STATO: CONTINUO" oppure "STATO: FINITO".

    Scegli lo stato in base alla FASE in cui ti trovi:

    ========================================================================
    === FASE 1: PIANIFICAZIONE EDITORIALE (PLANNER)                      ===
    ========================================================================
    Sei in questa fase quando stai scegliendo i piatti da inserire nel piano.

    * Usa "STATO: CONTINUO" se:
      - Devi ancora proporre dei topic.
      - Un topic è stato BLOCCATO dal controllo storico e devi inventare una variante.
      - Non hai ancora raggiunto il totale dei 3 piatti approvati.

    * Usa "STATO: FINITO" SOLO se:
      - Hai trovato ESATTAMENTE 3 topic diversi e TUTTI hanno ricevuto
        la risposta "OK" dal controllo storico.

    ========================================================================
    === FASE 2: RICERCA E FUSIONE DATI (RESEARCH)                        ===
    ========================================================================
    Sei in questa fase quando stai raccogliendo ingredienti e procedimento
    per scrivere l'articolo di un singolo piatto.

    * Usa "STATO: CONTINUO" se:
      - Hai trovato la ricetta principale, ma hai identificato che utilizza
        delle sottoricette (es. Ragù, Besciamella, Brodo).
      - Devi usare il tool `cerca_ricetta_nel_db` per recuperare la nostra
        versione locale di quella sottoricetta.
      - I dati trovati finora sono insufficienti o nulli.

    * Usa "STATO: FINITO" SOLO se:
      - Hai raccolto la ricetta principale in modo esaustivo.
      - Hai GIÀ recuperato dal DB locale TUTTE le eventuali sottoricette necessarie.
      - Hai tutti i dati completi e sei pronto per far scrivere il post al Writer.

    Args:
        reflection: Il testo del tuo ragionamento che spiega cosa hai capito
                    dai dati appena letti e che termina con lo stato corretto.

    Returns:
        Conferma di registrazione per il sistema di routing.
    """
    print("[think_tool] Riflessione ricevuta")
    return f"Riflessione registrata con successo: {reflection}"
