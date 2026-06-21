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

    FASE 1: PIANIFICAZIONE EDITORIALE (PLANNER)
    Sei in questa fase quando stai scegliendo i piatti da inserire nel piano.

    * Usa "STATO: CONTINUO" se:
      - Devi ancora proporre dei topic.
      - Un topic è stato BLOCCATO dal controllo storico e devi inventare una variante.
      - Non hai ancora raggiunto il totale dei 3 piatti approvati.

    * Usa "STATO: FINITO" SOLO se:
      - Hai trovato ESATTAMENTE 3 topic diversi e TUTTI hanno ricevuto
        la risposta "OK" dal controllo storico.

    FASE 2: ALGORITMO DI RICERCA RICORSIVA (RESEARCH)
    Obiettivo: Raccogliere la ricetta principale e risolvere ricorsivamente l'albero di tutte le sue sottoricette.

    * Usa "STATO: CONTINUO" se ti trovi in ALMENO UNO di questi scenari:
      - Stai attualmente eseguendo il PASSO 1 (DB locale) o il PASSO 2 (Web) per il topic principale o per una qualsiasi sottoricetta.
      - Una ricerca ha fallito e devi ottimizzare la query per effettuare un nuovo tentativo.
      - Hai rilevato una sottoricetta complessa che manca nel DB locale E hai accertato che il documento web corrente NON la spiega in modo esaustivo (dosi o procedimento mancanti), richiedendo quindi una ricerca web dedicata.

    * Usa "STATO: FINITO" SOLO E RIGOROSAMENTE se hai soddisfatto TUTTI questi requisiti:
      - 1. Hai recuperato con successo i dati per il topic principale.
      - 2. Hai analizzato ogni singola sottoricetta emersa e l'hai dichiarata "RISOLTA" applicando correttamente i criteri:
           - È presente nella nostra versione ufficiale nel DB locale, OPPURE
           - Il DB ha fallito, ma il documento web corrente la spiega già in modo completo con dosi e procedimento ("Tutto-in-uno"), OPPURE
           - Il documento web corrente era incompleto, ma l'hai cercata e recuperata con successo tramite una ricerca web dedicata.
      - 3. L'intero albero delle dipendenze è risolto e non ci sono più sottoricette pendenti o rami da esplorare.
      - 4. Non hai più nessun tool di ricerca da invocare.

    FASE 3: VALIDATORE E RISOLUTORE DI DIPENDENZE (SCORING)
    Obiettivo: Eleggere la Ricetta Madre, estrarre le sue dipendenze e validare i restanti documenti a cascata.

    REGOLA DI FERRO: In questa fase TI È SEVERAMENTE VIETATO usare "STATO: CONTINUO".
    Devi eseguire tutto il ragionamento in un'unica volta e concludere SEMPRE con "STATO: FINITO".

    Concludi il tuo ragionamento con "STATO: FINITO" SOLO E RIGOROSAMENTE DOPO aver:
    1. Eletto la MIGLIORE "Ricetta Madre" (Score 1) e scartato le versioni inferiori (Score 0).
    2. Dichiarato quali "Sottoricette" essa richiede esplicitamente.
    3. Valutato a cascata tutti gli altri documenti, dando Score 1 SOLO alle sottoricette richieste dalla Ricetta Madre (applicando la Regola di Salvataggio se c'è un solo file per quella sottoricetta).
    4. Assegnato Score 0 a qualsiasi ricetta "orfana" o fuori tema.
    5. Scritto fisicamente l'elenco esatto con l'ID di TUTTI i documenti e il relativo Score. Non puoi tralasciarne nessuno.

    REGOLA FONDAMENTALE:
    Durante il ragionamento rispetta l'ordine preciso dei documenti. Saltare o sbagliare documento è INAMMISSIBILE.

    Args:
        reflection: Il testo del tuo ragionamento che spiega cosa hai capito
            dai dati appena letti e che termina con lo stato corretto.

    Returns:
        Conferma di registrazione per il sistema di routing.
    """
    print("[think_tool] Riflessione ricevuta")
    return f"Riflessione registrata con successo: {reflection}"
