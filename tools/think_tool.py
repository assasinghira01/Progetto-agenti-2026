from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ThinkInput(BaseModel):
    analisi_contesto: str = Field(
        description="Analisi discorsiva e profonda dei dati appena ricevuti."
    )
    valutazione_opzioni: str = Field(
        description="Spiegazione dettagliata delle opzioni disponibili e dei ragionamenti scartati."
    )
    decisione_finale: str = Field(
        description="La mossa che hai deciso di fare, terminando con STATO: CONTINUO o STATO: FINITO."
    )


@tool(args_schema=ThinkInput)
def think_tool(
    analisi_contesto: str, valutazione_opzioni: str, decisione_finale: str
) -> str:
    """Tool FONDAMENTALE per la riflessione strategica e decisionale dell'agente.

    Usa questo tool obbligatoriamente dopo aver raccolto informazioni per spiegare
    il tuo ragionamento logico e dichiarare la tua prossima mossa al sistema.

    REGOLE DI FORMATTAZIONE OBBLIGATORIE:
    Il campo 'decisione_finale' DEVE terminare SEMPRE con una di queste due stringhe:
    "STATO: CONTINUO" oppure "STATO: FINITO".

    Scegli lo stato in base alla FASE in cui ti trovi:

    FASE 1: PIANIFICAZIONE EDITORIALE (PLANNER)

        Sei in questa fase quando stai scegliendo i piatti da inserire nel piano, partendo
        da una direttiva utente oppure da una pianificazione automatica.

        REGOLE FONDAMENTALI: Il numero di topic da approvare dipende dal contesto:
            - Se il piano è una PIANIFICAZIONE AUTOMATICA (input "PIANIFICAZIONE_AUTOMATICA"):
            devi ottenere ESATTAMENTE 3 topic approvati.
            - Se stai lavorando su un POST SINGOLO (qualsiasi altro input):
            devi ottenere ESATTAMENTE 1 topic approvato.
            - Il tuo compito è solo pianificare. Non DEVI MAI effettuare ricerche sul db e sul web. utilizza solo i dati
              in tuo possesso

        UTILIZZO DI "STATO: CONTINUO"
        Usa sempre "STATO: CONTINUO" se ti trovi in una di queste situazioni:

            1. Hai appena iniziato e non hai ancora nessun topic approvato.
            2. Hai proposto un topic ma non hai ancora ricevuto la risposta del tool
            `controlla_storico_post`.
            3. Il tool ha risposto "BLOCCATO" per un topic. In questo caso devi:
            - Nel caso di POST SINGOLO: applicare la procedura RAG (chiamare
                `get_ingredienti`, ragionare nel `think_tool`, poi proporre una variante
                o una ricetta simile).
            - Nel caso di PIANIFICAZIONE AUTOMATICA: scartare l'idea e generare un
                nuovo topic, sempre verificandolo con `controlla_storico_post`.
            4. Durante una MODIFICA o una RIGENERAZIONE con blacklist, finché non hai
            trovato il/i nuovo/i topic approvato/i richiesti.
            5. Quando stai ancora contando i topic: per la pianificazione automatica
            hai meno di 3 OK, per il post singolo hai 0 OK.

        CONTEGGIO OBBLIGATORIO NEL THINK_TOOL (per MODIFICHE e RIGENERAZIONI):
            - In ogni `think_tool` durante una modifica o rigenerazione, DEVI scrivere
            esplicitamente il conteggio:
            "Topic mantenuti: X. Nuovi topic approvati: Y. Totale: Z".
            - Appena il Totale Z raggiunge il numero richiesto (3 per pianificazione
            automatica, 1 per post singolo), DEVI passare immediatamente a STATO: FINITO.

        UTILIZZO DI "STATO: FINITO"
        Usa "STATO: FINITO" SOLO E RIGOROSAMENTE quando:

            1. Hai ESATTAMENTE il numero di topic approvati previsto per il tuo contesto:
            - 3 topic diversi per PIANIFICAZIONE AUTOMATICA.
            - 1 topic per POST SINGOLO.
            2. TUTTI questi topic hanno ricevuto esplicitamente la risposta "OK" dal tool
            `controlla_storico_post`.
            3. Non hai più alcun tool da chiamare relativamente alla pianificazione
            (hai già chiamato `get_ultimi_post` una volta se previsto, e hai verificato
            ogni nuovo topic).
            4. Nel caso di MODIFICA: hai integrato correttamente i topic mantenuti e il
            nuovo topic approvato, e il totale corrisponde esattamente al numero
            richiesto.
            5. Nel caso di RIGENERAZIONE con blacklist: hai proposto solo topic che NON
            sono nella blacklist, e tutti sono stati approvati.

        REGOLE SPECIALI PER IL FEEDBACK:
            - Se il feedback contiene "rigenera_con_blacklist:" o "rigenera", all’inizio
            del nuovo ciclo DEVI dichiarare nel `think_tool` che eviterai i topic
            presenti nella blacklist, e solo dopo inizi a proporre nuove idee.
            - Se il feedback contiene "modifica:", devi analizzare le istruzioni,
            identificare quali topic tenere e quali sostituire, e cercare ESATTAMENTE
            il numero di nuovi topic necessario per completare il totale.
            - In caso di modifica, NON ricontrollare i topic che hai deciso di mantenere:
            usa `controlla_storico_post` solo per i NUOVI topic.


    FASE 2: ALGORITMO DI RICERCA RICORSIVA (RESEARCH)

        **Obiettivo**: Raccogliere dati sulla ricetta principale e risolvere ricorsivamente l'intero albero delle dipendenze.


        #### REGOLA FONDAMENTALE DELLA MULTI-RICETTA
        Se la ricerca produce più documenti validi per lo stesso topic (es. più versioni dal web), ognuno è una **Ricetta Madre indipendente** il tuo compito
        NON è scegliere la migliore MA REPERIRE DATI PER OGNUNA DI ESSA.
        - Completa l'intero albero (sottoricette e loro dipendenze) per la PRIMA Ricetta Madre.
        - Poi passa alla SECONDA, e così via.
        - Non fermarti finché non hai processato TUTTE le Ricette Madri trovate.

        ####  GESTIONE DELLE QUERY ESPANSE
        - Per OGNI nuovo elemento (Ricetta Madre o sottoricetta), chiama `get_ingredienti` e `get_claim_per_retrieval` UNA VOLTA.
        - Usa il risultato per costruire una query espansa per `cerca_ricetta_nel_db`.
        - Se stai ritentando lo STESSO elemento, riutilizza la query senza chiamare di nuovo `get_ingredienti` e `get_claim_per_retrieval`.


        #### REGOLE
        **DIVIETO DI FUSIONE**: Se trovi più ricette, NON UNIRLE MAI. Ognuna è una Ricetta Madre indipendente.
        **REGOLA ANTI-LOOP**: Se una ricerca fallisce sia in DB che in Web, dichiara "FALLIMENTO: [Nome]" e ABBANDONA il ramo. NON RIPETERE.
        **REGOLA DI RICERCA**: Una volta trovate le ricette madri non eseguire nuovamenta la loro ricerca.
        **PROCESSAMENTO MULTI-RICETTA**: Se trovi più versioni, processale UNA ALLA VOLTA. Completa l'intero albero della prima, poi passa alla seconda.
        ""SOTTORICETTA**: Ricette madri molto semplici non vanno considerate come sottoricette.

        ####  UTILIZZO DI "STATO: CONTINUO"
        Usa "STATO: CONTINUO" in TUTTI questi scenari:

        1. Sei nella MACRO-FASE 1 e stai ancora cercando o valutando i dati per una qualsiasi Ricetta Madre (DB o Web).
        2. Stai per iniziare la MACRO-FASE 2 su una Ricetta Madre e devi elencare le sottoricette necessarie estratte dal testo.
        3. Sei nella MACRO-FASE 3 (Loop Sottoricette) e stai eseguendo i controlli (Check 1: DB, Check 2: Cortocircuito, Check 3: Web) per una sottoricetta.
        4. Un controllo è fallito e devi dichiarare esplicitamente il passaggio al controllo successivo (es. "CORTOCIRCUITO FALLITO, procedo al web").
        5. Hai appena scoperto una sottoricetta annidata e devi iniziare il loop su di essa.
        6. Hai completato l'albero di una Ricetta Madre ma esiste un'altra Ricetta Madre ancora da processare. Dichiara: "Ricetta Madre X completata. Passo alla Ricetta Madre Y. STATO: CONTINUO".


        #### UTILIZZO DI "STATO: FINITO"

        Usa "STATO: FINITO" SOLO quando hai soddisfatto TUTTI questi requisiti:

        1. Hai acquisito con successo TUTTE le Ricette Madri individuate per il topic.
        - *Nota*: Se dopo aver esplorato DB e web non trovi altre versioni, considera la lista chiusa.
        2. Per OGNI Ricetta Madre hai esaminato il testo ed estratto l'albero delle dipendenze (sottoricette).
        3. Ogni singola sottoricetta di OGNI Ricetta Madre è stata dichiarata "RISOLTA" (tramite DB, Cortocircuito o Web).
        4. L'intero albero delle dipendenze di TUTTE le Ricette Madri è chiuso, senza rami pendenti.
        5. Non hai più NESSUN tool da invocare: non ci sono altre ricette madri da iniziare né sottoricette da risolvere.


        ####  REGOLA ANTI-LOOP E ABBANDONO RAMO
        Se una sottoricetta fallisce TUTTI e TRE i Check (DB, Cortocircuito, Web):
        - Dichiara nel `think_tool`: "RAMO ABBANDONATO per [nome sottoricetta]".
        - Considera quel ramo chiuso forzatamente.
        - **NON ritentare all'infinito** – passa alla alla prossima sottoricetta o Ricetta Madre.



    FASE 3: VALIDATORE E RISOLUTORE DI DIPENDENZE (SCORING)

        Obiettivo: Eleggere la Ricetta Madre, estrarre le sue dipendenze e validare
        i restanti documenti a cascata, producendo un verdetto completo e immodificabile.

        REGOLA DI FERRO: In questa fase TI È SEVERAMENTE VIETATO usare "STATO: CONTINUO".
        Devi eseguire tutto il ragionamento in un'unica volta e concludere SEMPRE con
        "STATO: FINITO". Questa è la tua unica interazione: non ci sarà un prossimo turno.

        Concludi il tuo ragionamento con "STATO: FINITO" SOLO E RIGOROSAMENTE DOPO aver
        svolto, all'interno dei tre campi del think_tool, le seguenti operazioni
        nell'ordine indicato:

        ⚠️ COERENZA CON I CLAIM DEL KNOWLEDGE GRAPH
        Se nel contesto della validazione ti vengono forniti dei **CLAIM PERTINENTI DAL
        KNOWLEDGE GRAPH**, usali come ulteriore criterio di qualità e coerenza.
        - Confronta i documenti con i claim: se un documento approvato contraddice un
          claim già pubblicato sul blog, segnalalo esplicitamente nell'analisi e motiva
          perché lo accetti comunque o lo scarti.
        - Un conflitto grave (es. ingrediente principale diverso, tecnica opposta) deve
          influenzare il punteggio (Score) del documento o, se insanabile, portare alla
          sua esclusione.
        - Se non ci sono claim, procedi come al solito.

        ▶ NEL CAMPO 'analisi_contesto' (FASE 1 & 2):
            1. Eleggi la MIGLIORE "Ricetta Madre" tra tutti i documenti che parlano
                del topic principale. Assegna a questa Score 1. Assegna Score 0 a tutte
                le altre versioni della stessa ricetta (duplicati inferiori).
            2. Analizza ESCLUSIVAMENTE il procedimento della Ricetta Madre eletta ed
                estrai l'elenco delle sottoricette strettamente necessarie per realizzarla
                (es. besciamella, ragù). Non elencare mai gli ingredienti, solo le
                preparazioni derivate.
            3. Se sono presenti CLAIM PERTINENTI DAL KNOWLEDGE GRAPH, confrontali con
                la Ricetta Madre scelta: segnala eventuali conflitti e spiega perché
                la ricetta rimane valida o deve essere declassata

        ▶ NEL CAMPO 'valutazione_opzioni' (FASE 3):
            4. Prendi in esame TUTTI i documenti forniti, anche quelli che hai già
                esaminato nella FASE 1. Devi esaminarli UNO PER UNO, seguendo l'ordine
                esatto degli ID (DB_DOC_0, DB_DOC_1, ..., WEB_DOC_0, ...).
                Per ciascun documento:
                - Se è una sottoricetta necessaria (secondo quanto emerso dalla Ricetta
                Madre), assegna Score 1 al migliore (in caso di duplicati) e
                Score 0 agli altri duplicati inferiori.
                - Se il documento è irrilevante, fuori tema o non serve, assegna Score 0
                con motivazione esplicita.
            5. **NON PUOI SALTARE NESSUN DOCUMENTO**. Ogni ID che ti è stato fornito
                deve comparire nella valutazione e poi nell'elenco finale.

        ▶ NEL CAMPO 'decisione_finale' (VERDETTO):
            6. Scrivi l'elenco FISICO ED ESATTO di TUTTI i documenti valutati,
                uno per riga, nel formato obbligatorio:
                ID_DOC [TITOLO]: Score X - Motivo: ...
            7. L'elenco DEVE contenere ESATTAMENTE il numero di documenti che ti sono
                stati forniti. Non puoi ometterne nessuno.
            8. Rispetta rigorosamente l'ordine e gli ID originali con cui i documenti
                ti sono stati presentati (es. DB_DOC_0, DB_DOC_1, WEB_DOC_0, WEB_DOC_1).
            9. **Prima di scrivere "STATO: FINITO", conta le righe del tuo elenco e
                verifica che corrispondano al numero totale di documenti indicato all'inizio.**
            10. Subito dopo l'ultima riga, scrivi "STATO: FINITO".

        RICORDA: Omettere un documento è un ERRORE CRITICO. Anche i documenti con
        Score 0 devono apparire nella lista.
        Non usare mai "STATO: CONTINUO" in questa fase. dimmi i punti dove inserire le correzioni


    Args:
        analisi_contesto: La tua analisi dei dati.
        valutazione_opzioni: Il tuo ragionamento sulle opzioni.
        decisione_finale: La tua decisione che termina con lo stato.

    Returns:
        Conferma di registrazione per il sistema di routing.
    """

    reflection_completa = (
        f"--- ANALISI ---\n{analisi_contesto}\n\n"
        f"--- OPZIONI ---\n{valutazione_opzioni}\n\n"
        f"--- DECISIONE ---\n{decisione_finale}"
    )
    print("\n[think_tool] Riflessione strutturata ricevuta!")
    return f"Riflessione registrata con successo:\n{reflection_completa}"
