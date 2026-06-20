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
    Sei in questa fase quando stai raccogliendo ingredienti e procedimento.

    * * Usa "STATO: CONTINUO" se:
      - Hai trovato la ricetta principale, ma hai notato ingredienti che rappressentano sottoricette come Maionese,
        Ragù, Brodo, Besciamella, ecc., e devi ancora cercarli.
      - I dati trovati finora sono insufficienti o stai per cambiare fonte.

    * Usa "STATO: FINITO" SOLO se:
      - Hai raccolto la ricetta principale in modo esaustivo.
      - Hai GIÀ cercato e  recuperato TUTTE le eventuali sottoricette necessarie.
      - Hai tutti i dati completi e Non devi cercare più nulla.


      ========================================================================
      FASE 3 - VALIDAZIONE E RE-RANKING (SCORING)
      ========================================================================
      Obiettivo: Valutare l'utilità di OGNI singola fonte recuperata (DB e Web).
      Usa "STATO: CONTINUO" se:
      - ci sono moltissimi documenti e hai bisogno di riflettere in più passaggi prima di dare i voti definitivi.
      Usa "STATO: FINITO" SOLO se:
      - hai analizzato mentalmente OGNI documento DB e WEB presente;
      - hai deciso quale punteggio (0=Irrilevante o inutile, 1=Fondamentale) merita ogni documento;
      - hai formulato un motivo conciso per ogni punteggio assegnato;
      - hai stabilito se i documenti approvati (score == 1) bastano per scrivere un articolo di alta qualità.

      REGOLA FONDAMENTALE

      Durante il ragionamento rispetta l'ordine preciso dei documenti, saltare o sbagliare documento è INAMISSIBILE

    Args:
        reflection: Il testo del tuo ragionamento che spiega cosa hai capito
                    dai dati appena letti e che termina con lo stato corretto.

    Returns:
        Conferma di registrazione per il sistema di routing.
    """
    print("[think_tool] Riflessione ricevuta")
    return f"Riflessione registrata con successo: {reflection}"
