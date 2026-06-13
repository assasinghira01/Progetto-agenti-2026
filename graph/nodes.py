from langgraph.types import Command
from langchain_core.messages import (
    RemoveMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    filter_messages,
)
from langgraph.types import interrupt
from config import llm, llm_con_tools
from graph.schemas import (
    RecipeDraft,
    TopicExtraction,
    ValidationResult,
)
from graph.state import Blog_Cucina
from knowledge_graph.neo4j_manager import kg_client

# per il planner
llm_structured = llm.with_structured_output(TopicExtraction)


def planner_node(state: Blog_Cucina):

    print("\n--- [NODO 1: PLANNER (LLM)] ---")
    input = state["input_utente"]
    messaggi = state["messages"]
    feedback = state.get("human_feedback")

    if input == "PIANIFICAZIONE_AUTOMATICA":

        testo_prompt = """
        # Ruolo
    Sei un planner editoriale per un blog di cucina dove vengono pubblicati dei post sulla preparazione di piatti. 
    
    # REGOLE DI RAGIONAMENTO (OBBLIGATORIE):
    Il tuo flusso di lavoro deve essere rigorosamente questo:
    1. Chiama un tool (es. get_ultimi_post o controlla_storico_post).
    2. APPENA RICEVI LA RISPOSTA DEL TOOL, prima di fare qualsiasi altra cosa o chiamare altri tool di ricerca, DEVI chiamare il `think_tool`.
    3. Nel `think_tool`, spiega cosa hai appena scoperto e decidi qual è il tuo prossimo passo logico.
    4. Ripeti questo ciclo (Ricerca -> Think -> Ricerca -> Think) finché non hai 3 topic approvati. Non chiamare tool di ricerca in sequenza senza usare il think_tool in mezzo.
    5. motiva tutto cio che fai nel think_tool, è fondamentale per la tua capacità di pianificare in modo intelligente e strategico.
    
    
    ## FASI DI LAVORO (Seguile in questo ordine esatto):

    FASE 1 - RECUPERO STORICO: 
    Chiama il tool `get_ultimi_post` per capire cosa è stato pubblicato di recente. (Non richiamarlo più di una volta). Usa il `think_tool` per riflettere sui risultati.

    FASE 2 - IDEAZIONE INTERNA E VERIFICA (IL CUORE DEL TUO LAVORO):
    NON generare ancora il piano finale. Nella tua mente, elabora  SOLO 3 nuovi topic (variando le categorie, se sono state pubblicati due primi e una secondo puoi generare un antipasto o un dessert, se una 
    tipolgia non ha publicata dai priorita a quest'ultima, se non sono stati pubblicati piati vegani  dai priorita ad un piatto vegano. L'obbiettivo è creare contenuti unici e rilevanti per i lettori del blog non annoiandoli).
    Per **OGNUNO** di questi 3 topic, è ASSOLUTAMENTE OBBLIGATORIO chiamare il tool `controlla_storico_post` SU ognuno di essi, SEI OBBLIGATO A FARLO A PRESCINDERE. 
    - Se il tool risponde "BLOCCATO", scarta l'idea, usa il `think_tool` per riflettere sull'errore, e proponi un piatto diverso verificandolo di nuovo.
    - Se il tool risponde "OK", il topic è approvato.

    FASE 3 - GENERAZIONE DEL PIANO:
    SOLO DOPO aver ottenuto 3 risposte "OK" dal tool `controlla_storico_post`, puoi procedere a generare il testo finale del piano editoriale con i 3 topic approvati.

    ## Formato dei topic (REGOLA FERREA)
    - **Singola Preparazione:** Il topic deve essere una ricetta reale, specifica e SINGOLA. 
    - È ASSOLUTAMENTE VIETATO combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
    - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
    - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
    - ✅ CORRETTO: "Pollo al limone"
    - ✅ CORRETTO: "Verdure grigliate"
    - ✅ CORRETTO: "Filetto di manzo al pepe verde"
    - NON cercare dati online o in locale. Il tuo compito è solo pianificare.
    
    """

        if feedback and feedback.startswith("rigenera_con_blacklist:"):

            lista_nera = feedback.split(":")[1]

            print(
                f"   [Planner] Rilevato feedback 'rigenera'. Topic bloccati: {lista_nera}"
            )

            testo_prompt += f"""
            
            --- FEEDBACK UTENTE (PIANO RIFIUTATO) ---
            ATTENZIONE: Il piano editoriale precedente è stato RIFIUTATO. 
            L'utente ha esplicitamente scartato i seguenti topic: {lista_nera}.
            
            REGOLA ASSOLUTA: NON PROPORRE NESSUNO DI QUESTI TOPIC SCARTATI. 
            Devi generare 3 idee COMPLETAMENTE DIVERSE.
            
            --- REGOLE DI WORKFLOW PER LA RIGENERAZIONE ---
            1. Nel tuo PRIMO utilizzo del `think_tool` in questo nuovo ciclo, DEVI dichiarare esplicitamente di aver ricevuto il feedback negativo e menzionare i topic scartati ({lista_nera}) che eviterai.
            2. Chiama il tool `get_ultimi_post` per capire cosa è stato pubblicato di recente. (Non richiamarlo più di una volta).
            3. Continua a usare il `think_tool` dopo ogni controllo con `controlla_storico_post`. 
            4. Usa la dicitura "STATO: CONTINUO" finché non trovi 3 nuove idee approvate dal tool.
            5. Usa la dicitura "STATO: FINITO" solo alla fine, quando hai i 3 topic definitivi.
            
            --- FORMATO DEI TOPIC (REGOLA FERREA) ---
            - Singola Preparazione: Il topic deve essere una ricetta reale, specifica e SINGOLA. 
            - È ASSOLUTAMENTE VIETATO combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
              - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
              - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
              - ✅ CORRETTO: "Pollo al limone"
              - ✅ CORRETTO: "Verdure grigliate"
              - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            
            [VINCOLO TASSATIVO]: NON cercare dati online o in locale. Il tuo compito è ESCLUSIVAMENTE pianificare internamente.
        
                        """

        if feedback and feedback.startswith("modifica:"):

            parti = feedback.split("|")
            istruzioni = parti[0].replace("modifica:", "").strip()

            piano_salvato = ""
            blacklist_salvata = ""

            for p in parti:
                if p.startswith("piano:"):
                    piano_salvato = p.replace("piano:", "").strip()
                elif p.startswith("blacklist:"):
                    blacklist_salvata = p.replace("blacklist:", "").strip()

            print(
                f"   [Planner] Rilevato feedback 'modifica'. Istruzioni: {istruzioni}, {piano_salvato}, {blacklist_salvata}"
            )
            testo_prompt += f"""
            
            --- FEEDBACK UTENTE (RICHIESTA DI MODIFICA) ---
            L'utente ha richiesto le seguenti MODIFICHE SPECIFICHE al piano: 
            "{istruzioni}"
            
            IL PIANO PRECEDENTE ERA COMPOSTO DA QUESTI 3 TOPIC: {piano_salvato}
            
            REGOLA MATEMATICA E LOGICA ASSOLUTA:
            1. Analizza la richiesta dell'utente: identifica QUALE topic deve essere eliminato e COSA l'utente vuole al suo posto (presta estrema attenzione se chiede una specifica categoria come "contorno", "primo", "pesce", ecc.)
            devi quindi basarti su "{istruzioni}". REGOLA NON DEVI MAI COP
            2. MANTIENI INTATTI E NON CANCELLARE gli altri topic del piano precedente che non sono stati nominati per l'eliminazione.
            3. Il NUOVO topic generato DEVE RISPETTARE ALLA LETTERA la categoria o l'ingrediente richiesto dall'utente nella modifica.
            4. Alla fine del processo, il piano finale DEVE contenere rigorosamente: i topic vecchi mantenuti + il nuovo topic approvato (Totale esatto: 3 topic).
            """

            if blacklist_salvata:
                testo_prompt += f"""
            Inoltre, ricorda che nei cicli precedenti l'utente aveva GIA' SCARTATO questi topic: {blacklist_salvata}. 
            NON DEVI assolutamente riproporli.
            """

            if blacklist_salvata:
                testo_prompt += f"""
            Inoltre, ricorda che nei cicli precedenti l'utente aveva GIA' SCARTATO questi topic: {blacklist_salvata}. 
            NON DEVI assolutamente riproporli.
            """

            testo_prompt += """
            --- REGOLE DI WORKFLOW PER LA MODIFICA ---
            1. Nel tuo PRIMO `think_tool`, dichiara: quali topic tieni, quale elimini, e QUANTI nuovi topic devi cercare (es. "Devo cercare 1 nuovo topic").
            2. Usa il tool `get_ultimi_post` per controllare i duplicati recenti.
            3. Usa `controlla_storico_post` SOLO per verificare i NUOVI topic che stai introducendo.
            4. 🚨 REGOLA DI STOP (IMPORTANTISSIMA): In ogni `think_tool`, DEVI FARE IL CONTEGGIO ESPLICITO. Scrivi letteralmente: "Topic mantenuti: X. Nuovi topic approvati: Y. Totale: Z". 
            5. Appena il tuo Totale Z arriva a 3, DEVI FERMARTI IMMEDIATAMENTE e chiudere il messaggio con "STATO: FINITO". È severamente vietato cercare un quarto topic.
                        
            
            
            [VINCOLO TASSATIVO]: NON cercare dati online o in locale. Il tuo compito è ESCLUSIVAMENTE pianificare internamente.
            """

        prompt = SystemMessage(content=testo_prompt)

    else:
        prompt = SystemMessage(content=f"""
        # Ruolo
    Sei un planner editoriale per un blog di cucina. Il tuo compito in questa sessione è valutare e validare una SINGOLA proposta di ricetta inviata dall'utente. 
    
    # REGOLE DI RAGIONAMENTO (OBBLIGATORIE):
    Il tuo flusso di lavoro deve essere rigorosamente questo:
    1. Chiama un tool (es. controlla_storico_post o proponi_variante).
    2. APPENA RICEVI LA RISPOSTA DEL TOOL, prima di fare qualsiasi altra cosa, DEVI chiamare il `think_tool`.
    3. Nel `think_tool`, spiega cosa hai appena scoperto e decidi qual è il tuo prossimo passo logico. Termina con "STATO: CONTINUO" se stai ancora lavorando.
    4. Ripeti questo ciclo (Ricerca/Azione -> Think -> Ricerca/Azione -> Think) finché non ottieni l'approvazione definitiva. Non chiamare tool in sequenza senza usare il think_tool in mezzo.
    5. Motiva tutto ciò che fai nel think_tool, è fondamentale per la tua capacità di pianificare in modo intelligente e strategico.
    
    
    ## FASI DI LAVORO (Seguile in questo ordine esatto):

    FASE 1 - VERIFICA DEL TOPIC PROPOSTO: 
    L'utente ha proposto una specifica ricetta. Usa immediatamente il tool `controlla_storico_post` su questa ricetta per verificare se ha già un post associato nel Knowledge Graph.
    - Se la risposta inizia con "OK": il topic è nuovo e va bene. Passa alla Fase 3.
    - Se la risposta inizia con "BLOCCATO": significa che il post è già stato pubblicato di recente. Passa alla Fase 2.

    FASE 2 - GESTIONE DEL BLOCCO E VARIANTE (SOLO SE NECESSARIO):
    Se il topic originale è bloccato, usa il `think_tool` per riflettere sull'errore. 
    Successivamente, DEVI chiamare il tool `proponi_variante` per generare una variante interessante e coerente con la richiesta iniziale dell'utente.
    Una volta ottenuta la variante, è ASSOLUTAMENTE OBBLIGATORIO ripetere il controllo chiamando di nuovo `controlla_storico_post` sulla nuova variante.

    FASE 3 - CONCLUSIONE DELLA PIANIFICAZIONE (REGOLA DI STOP ASSOLUTA):
    Appena ottieni un "OK" definitivo per il topic (sia esso l'originale proposto dall'utente o la tua variante approvata), LA FASE DI PIANIFICAZIONE È FINITA!!!!
    Devi OBBLIGATORIAMENTE fare un'ultima chiamata al `think_tool` in cui confermi il piatto e scrivi ESATTAMENTE queste parole: "Topic approvato. STATO: FINITO".
    È severamente vietato continuare a cercare, riflettere o proporre varianti dopo aver raggiunto l' "OK".

    ## Formato dei topic (REGOLA FERREA)
    - **Singola Preparazione:** Il topic (o la variante) deve essere una ricetta reale, specifica e SINGOLA. 
    - È ASSOLUTAMENTE VIETATO combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
    - NOMI SPECIFICI: NON USARE MAI nomi generici o nomi di categorie come titolo della ricetta. Devi usare il nome esatto del piatto.
      - ❌ SBAGLIATO (VIETATO): "Variante a base di pesce", "Un dolce al cioccolato", "Contorno di verdure"
      - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
      - ✅ CORRETTO: "Spaghetti alle vongole", "Caprese al limone", "Zucchine trifolate", "Pollo al limone"
    - NON cercare dati online o in locale. Il tuo compito è solo pianificare.
    
    """)

    if messaggi and isinstance(messaggi[-1], ToolMessage):
        risposta_llm = llm_con_tools.invoke([prompt] + messaggi)
        return {"messages": [risposta_llm], "nodo_chiamante": "planner"}

    if input == "PIANIFICAZIONE_AUTOMATICA":

        messaggi_da_inviare = [
            prompt,
            HumanMessage(content="Inizia la ricerca..."),
        ]

        risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)

        return {"messages": [risposta_llm], "nodo_chiamante": "planner"}

    else:
        input_utente = state["input_utente"]

        # 1. Estraiamo il topic principale

        risultato = llm_structured.invoke(
            [HumanMessage(content=f"Estrai il topic: {input_utente}")]
        )
        topic_estratto = risultato.topic.strip().capitalize()

        print(f" Topic identificato: {topic_estratto}")

        messaggi_da_inviare = [
            prompt,
            HumanMessage(
                content=f"Controlla se il topic '{topic_estratto}' ha già un post associato."
            ),
        ]

        risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)

        return {
            "messages": [risposta_llm],
            "topic_corrente": [topic_estratto],
            "nodo_chiamante": "planner",
        }


def human_review_planner(state: Blog_Cucina):
    print("\n--- [NODO: HUMAN REVIEW PLANNER] ---")

    feedback = interrupt({"msg": "In attesa di revisione umana"})

    print(f" Feedback ricevuto: {feedback}")
    piano = state.get("piano_editoriale")
    if feedback == "APPROVA":
        topic = piano.sequenza_post[0].topic
        return Command(
            update={
                "topic_corrente": [topic],
                "human_feedback": "approvato",
                "piano_editoriale": piano,
            },
            goto="research",
        )
    elif feedback == "RIGENERA":
        nuovi_scartati = [post.topic for post in piano.sequenza_post]

        # Recuperiamo l'eventuale blacklist precedente dallo stato
        vecchio_feedback = state.get("human_feedback", "")

        vecchia_blacklist = []
        if vecchio_feedback and vecchio_feedback.startswith("rigenera_con_blacklist:"):

            stringa_vecchi = vecchio_feedback.split(":")[1]

            vecchia_blacklist = [t.strip() for t in stringa_vecchi.split(",")]

        blacklist_totale = vecchia_blacklist + [
            t for t in nuovi_scartati if t not in vecchia_blacklist
        ]

        topic_scartati_aggiornati = ", ".join(blacklist_totale)

        # Cancelliamo la cronologia dei messaggi per fare spazio al nuovo ragionamento
        messaggi_da_cancellare = [RemoveMessage(id=m.id) for m in state["messages"]]

        return Command(
            update={
                "human_feedback": f"rigenera_con_blacklist:{topic_scartati_aggiornati}",
                "piano_editoriale": None,
                "messages": messaggi_da_cancellare,
            },
            goto="planner",
        )

    elif feedback == "MODIFICA":
        # Chiediamo all'utente cosa vuole cambiare

        istruzioni_modifica = input(
            "\n📝 Scrivi le tue istruzioni di modifica (es. 'Sostituisci il dolce con un antipasto'): "
        )

        # recuperiamo i topic precedenti
        piano_attuale = state.get("piano_editoriale")
        topic_attuali = (
            ", ".join([post.topic for post in piano_attuale.sequenza_post])
            if piano_attuale
            else ""
        )

        # Recuperiamo la blacklist precedente per non perderla
        vecchio_feedback = state.get("human_feedback", "")
        vecchia_blacklist = ""

        # Estraiamo la blacklist sia che provenga da un vecchio RIGENERA sia da un vecchia MODIFICA
        if "rigenera_con_blacklist:" in vecchio_feedback:
            vecchia_blacklist = vecchio_feedback.split("rigenera_con_blacklist:")[1]
        elif "|blacklist:" in vecchio_feedback:
            vecchia_blacklist = vecchio_feedback.split("|blacklist:")[1]

        # Costruiamo il nuovo feedback unendo istruzioni e vecchia blacklist
        nuovo_feedback = f"modifica:{istruzioni_modifica}|piano:{topic_attuali}"
        if vecchia_blacklist:
            nuovo_feedback += f"|blacklist:{vecchia_blacklist}"

        messaggi_da_cancellare = [RemoveMessage(id=m.id) for m in state["messages"]]

        return Command(
            update={
                "human_feedback": f"modifica:{nuovo_feedback}",
                "piano_editoriale": None,
                "messages": messaggi_da_cancellare,
            },
            goto="planner",
        )


def krag_research_node(state: Blog_Cucina):
    print("\n--- [NODO 2: RICERCA MCP (Agente Autonomo)] ---")
    topic = state["topic_corrente"]
    messaggi = state.get("messages", [])

    istruzioni_mcp = SystemMessage(content=f"""
    Sei l'agente investigatore del blog di cucina, nel nsotro blog parleremo di ricette. Il topic ATTUALE su cui devi lavorare è: ' '.
    
    REGOLE DI COMPORTAMENTO (rispettale rigorosamente in ordine):
    1. **Prima azione assoluta verifica che il post non sia gia stato pubblicato**: chiama 'controlla_storico_post' su '{topic}'.
    
       - Se inizia con "OK": significa che il post è nuovo. Devi raccogliere le informazioni seguendo questo esatto flusso di ragionamento (Thought -> Action):
           
           
            Sai che il nostro database locale contiene solo ricette classiche,  Effetua la ricerca su DB LOCALE usando 'cerca_ricetta_nel_db'. Controlla che nei dati estratti dal db  ci siano dati validi in base al '{topic}' in quel caso non effettuare la ricerca online. Nel caso in cui
            L'utente richieda varianti, modifiche o versioni light/leggere delle ricette devi cercare informazioni sul web utilizzando "esegui_ricerca_web" DEVI estrarre gli INGREDIENTI, le DOSI e il PROCEDIMENTO. Usa la ricerca web nel caso in cui i dati del DB locale siano insufficienti o non pertinenti al topic richiesto.
            

    3. Se ricevi "ERRORE" o "KRAG_ERRORE", segnala immediatamente il problema all'utente e chiedi istruzioni.
    """)

    if not messaggi:
        messaggi_da_inviare = [
            istruzioni_mcp,
            HumanMessage(content=f"Inizia la ricerca per il topic: {topic}"),
        ]
    else:
        messaggi_da_inviare = [istruzioni_mcp] + messaggi

    risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)

    return {"messages": [risposta_llm], "nodo_chiamante": "research"}


def validator_node(state: Blog_Cucina):
    print("\n--- [NODO 3: VALIDATORE (Fact-Checking Incrociato)] ---")

    topic = state["topic_corrente"]
    dati_db_locale = state.get("rag_documents", [])
    dati_web_grezzi = state.get("web_documents", [])

    # Pulizia e separazione dei blocchi web
    dati_web = []
    for blocco in dati_web_grezzi:
        if isinstance(blocco, str) and "--- Fonte Web:" in blocco:
            estratti = blocco.split("--- Fonte Web:")
            for estratto in estratti:
                if estratto.strip():
                    dati_web.append("--- Fonte Web:" + estratto)
        else:
            dati_web.append(blocco)

    # Numerazione dei documenti del DB Locale per l'LLM
    if dati_db_locale:
        db_numerati = []
        for idx, doc in enumerate(dati_db_locale):
            db_numerati.append(f"\n=== DB_DOC_{idx} ===\n{doc}\n")
        testo_db = "\n".join(db_numerati)
    else:
        testo_db = "NESSUNA RICETTA TROVATA NEL DB LOCALE"

    # Numerazione dei documenti web per l'LLM
    if dati_web:
        web_numerati = []
        for idx, doc in enumerate(dati_web):
            web_numerati.append(f"\n=== WEB_DOC_{idx} ===\n{doc}\n")
        testo_web = "\n".join(web_numerati)
    else:
        testo_web = "NESSUN DATO COLLATERALE DAL WEB"

    prompt = f"""
Analizza la fattibilità editoriale per il piatto: '{topic}'.

=== FONTE DI VERITÀ INTERNA (DB LOCALE) ===
{testo_db}

=== INFORMAZIONI DI CONTESTO (RICERCA WEB) ===
{testo_web}

COMPITO E CRITERI DI VALUTAZIONE:

1. SENSO GASTRONOMICO:
   Blocca ricette assurde o accostamenti privi di senso.
2. SUFFICIENZA DEI DATI:
   Verifica se abbiamo abbastanza informazioni per scrivere un articolo attendibile.
3. PERTINENZA DEL DB LOCALE:
   Identifica quali documenti del DB locale parlano ESATTAMENTE del topic '{topic}'.
   Inserisci gli indici numerici (es. 0, 1) nel campo 'documenti_db_approvati'. 
   Se contengono ricette completamente scollegate (es. cerchi Besciamella e trovi Pan di zenzero), IGNORALI.
4. ESITO:
   - True se il topic è valido e documentato.
   - False se il topic è assurdo oppure non esistono dati utilizzabili.

5. QUALITÀ FONTI WEB:
   I documenti web sono identificati come: WEB_DOC_0, WEB_DOC_1...
   Seleziona SOLO gli ID dei documenti migliori.
   DEVE essere scelto un solo documento. La scelta deve essere basata sulla pertinenza, autorevolezza e completezza.
   Inserisci gli ID nel campo: documenti_web_approvati
"""

    llm_validator = llm.with_structured_output(ValidationResult)
    esito = llm_validator.invoke([HumanMessage(content=prompt)])

    print(f" Esito Validazione: {esito.is_valid}")
    print(f" Motivazione: {esito.reasoning}")
    print(f" Usa DB Locale: {esito.usa_db_locale}")
    print(f" Documenti DB Approvati: {esito.documenti_db_approvati}")
    print(f" Documenti Web Approvati: {esito.documenti_web_approvati}")

    # =========================================================
    # PRUNING DELLO STATO (FILTRAGGIO MATEMATICO)
    # =========================================================

    # 1. Filtriamo i documenti WEB
    dati_web_filtrati = []
    for idx in esito.documenti_web_approvati:
        if 0 <= idx < len(dati_web):
            dati_web_filtrati.append(dati_web[idx])

    # 2. Filtriamo i documenti del DB LOCALE
    dati_db_filtrati = []

    if not esito.usa_db_locale:
        print("[VALIDATORE] DB locale non pertinente.")
    else:
        print("[VALIDATORE] Estraggo solo le ricette pertinenti...")

        for idx in esito.documenti_db_approvati:

            if 0 <= idx < len(dati_db_locale):

                doc = dati_db_locale[idx]

                # Caso mega-chunk con più ricette
                if "Ricetta:" in doc:

                    blocchi = doc.split("Ricetta:")

                    for blocco in blocchi:

                        if topic.lower().strip() in blocco.lower():

                            dati_db_filtrati.append("Ricetta: " + blocco.strip())

            else:
                # Documento singolo normale
                if topic.lower().strip() in doc.lower():
                    dati_db_filtrati.append(doc)

    direttiva = (
        f"Fonte web: {esito.documenti_web_approvati}. "
        f"Fonte DB: {esito.documenti_db_approvati}. "
        f"Motivazione: {esito.motivazione_qualita}"
    )

    print(f" Dati web finali passati al writer: {len(dati_web_filtrati)} documenti")
    print(f" Dati DB finali passati al writer: {len(dati_db_filtrati)} documenti")

    return {
        "is_valid": esito.is_valid,
        "valutazione_qualita": direttiva,
        "approved_web_documents": dati_web_filtrati,
        "approved_db_documents": dati_db_filtrati,
    }


# writer node: sintetizza le informazioni approvate e scrivi la bozza del post in markdown, con attenzione alla distinzione tra ingredienti diretti e sotto-ricette, e alla gerarchia degli ingredienti. Applica eventuali feedback umani ricevuti per correggere o migliorare la bozza prima di generare il markdown finale.
def writer_node(state: Blog_Cucina):
    print("\n--- [NODO 4: WRITER (Sintesi e Grounding)] ---")
    topic = state["topic_corrente"]
    dati_db_locale = state.get("rag_documents", [])
    dati_web = state.get("approved_web_documents", [])
    testo_db = "\n".join(dati_db_locale) if dati_db_locale else "NESSUN DATO IN LOCALE"
    testo_web = "\n".join(dati_web) if dati_web else "NESSUN DATO DAL WEB"
    feedback = state.get("human_feedback")
    print(f"datiweb: {testo_web}")
    print(f"datilocale: {testo_db}")
    istruzione_correzione = (
        f"""

FEEDBACK REDATTORE:

{feedback}

Applica queste modifiche.
"""
        if feedback
        else ""
    )

    prompt = f"""
Sei un food blogger professionista.

ARGOMENTO:
{topic}

Devi produrre una ricetta strutturata.

REGOLE IMPORTANTI:

- Non inventare ingredienti.
- Non inventare quantità.
- Non inventare preparazioni.
- Usa solo le informazioni presenti nelle fonti.
- INGREDIENTI: Estrai le dosi ESCLUSIVAMENTE dal testo della fonte selezionata.
 È severamente vietato unire le dosi di due siti diversi o inventarle. 
Rispetta la divisione gerarchica tra ingredienti diretti e sotto-ricette.


SOTTORICETTE:
Se individui preparazioni autonome (es. Ragù, Besciamella, Crema pasticcera, Ganache, Pastella)
NON inserirle negli ingredienti diretti.
Crea invece una SottoRicetta con:
- nome_specifico
- classe_astratta
- ingredienti
Esempio:
nome_specifico = "Ragù per arancini"
classe_astratta = "Ragù"
ingredienti = [...]
REGOLA TASSATIVA ANTI-TOPPING E CONDIMENTI 
È severamente VIETATO creare una SottoRicetta per raggruppamenti di ingredienti crudi o pronti che devono solo essere posizionati sopra il piatto.
Se la fonte ha un titolo come "PER CONDIRE", "TOPPING", "PER GUARNIRE", "FARCITURA" (es. pomodoro, mozzarella, prosciutto su una pizza, o verdure in un'insalata):
1. DEVI IGNORARE QUEL TITOLO.
2. NON CREARE ALCUNA SOTTORICETTA.
3. Prendi tutti quegli ingredienti e inseriscili nella lista degli INGREDIENTI DIRETTI, assegnando loro la fase_utilizzo "Condimento" o "Guarnizione"

INGREDIENTI DIRETTI:
Inserisci qui soltanto gli ingredienti che appartengono direttamente alla ricetta principale.
Esempio:
Arancini:
- Riso
- Burro
- Zafferano

INTRODUZIONE:
max 30 parole.
PREPARAZIONE:
max 100 parole.

{istruzione_correzione}

FONTI:

=== DB LOCALE ===
{testo_db}

=== WEB ===
{testo_web}
"""

    llm_writer = llm.with_structured_output(RecipeDraft)
    draft = llm_writer.invoke([HumanMessage(content=prompt)])
    # =======================================================
    # PULIZIA DATI "ANTI-MATRIOSKA" (Prima di generare il Markdown)
    # =======================================================
    sottoricette_pulite = []
    ingredienti_da_spostare = []
    for sub in draft.sotto_ricette:
        if not sub.ingredienti:
            continue
        if len(sub.ingredienti) == 1:
            ing_nome = sub.ingredienti[0].nome.lower()
            sub_nome = sub.classe_astratta.lower()
            # Se la preparazione contiene se stessa come ingrediente
            if sub_nome in ing_nome or ing_nome in sub_nome:
                # Spostiamo l'ingrediente finito nella lista dei diretti
                ingredienti_da_spostare.append(sub.ingredienti[0])
                continue  # Ignora la sottoricetta

        # Se passa i controlli, la conserviamo
        sottoricette_pulite.append(sub)
    # Aggiorniamo l'oggetto draft originale con i dati puliti
    draft.sotto_ricette = sottoricette_pulite
    draft.ingredienti_diretti.extend(ingredienti_da_spostare)
    # =======================================================
    print("\n===== DEBUG WRITER =====")
    print(draft.model_dump())
    print("========================\n")
    markdown = f"# {draft.titolo}\n\n"
    markdown += "## Introduzione\n\n"
    markdown += draft.introduzione + "\n\n"
    markdown += "## Ingredienti Principali\n\n"
    for ing in draft.ingredienti_diretti:
        markdown += f"- {ing.nome}: " f"{ing.quantita}\n"
    if draft.sotto_ricette:
        markdown += "\n"
        for sub in draft.sotto_ricette:
            markdown += f"### {sub.nome_specifico}\n\n"
            for ing in sub.ingredienti:
                markdown += f"- {ing.nome}: " f"{ing.quantita}\n"
            markdown += "\n"
    markdown += "\n## Preparazione\n\n" f"{draft.preparazione}\n\n"
    markdown += "## Fonte\n\n" f"{draft.fonte}"
    return {
        "recipe_draft": draft,
        "post_draft": markdown,
    }


def human_review_node(state: Blog_Cucina):
    print("\n--- [NODO 5: HUMAN-IN-THE-LOOP (Approvazione)] ---")
    bozza = state.get("post_draft", "")

    print("\n================ BOZZA DEL POST ================\n")
    print(bozza)
    print("\n================================================\n")

    feedback = interrupt(
        "Bozza pronta! Digita 'Approvo' per pubblicare, o scrivi le modifiche (es. 'Mettici meno sale')."
    )

    print(f" Hai risposto: {feedback}")
    # Salviamo la tua risposta nello stato
    return {"human_feedback": feedback}


def kg_update_node(state: Blog_Cucina):
    print("\n--- [NODO 6: KG UPDATE (Aggiornamento Memoria)] ---")

    draft = state.get("recipe_draft")
    if not draft:
        print("[ERRORE] Nessun recipe_draft trovato nello stato.")
        return {}

    topic_finale = state["topic_corrente"]
    fonte = draft.fonte

    # ==========================================
    # 1. ESTRAZIONE INGREDIENTI DIRETTI
    # ==========================================
    ingredienti_diretti = [
        {
            "nome": ing.nome,
            "quantita": ing.quantita,
            "fase_utilizzo": getattr(ing, "fase_utilizzo", "Base"),
        }
        for ing in draft.ingredienti_diretti
    ]

    # ==========================================
    # 2. ESTRAZIONE SOTTO RICETTE
    # ==========================================
    sotto_ricette = [
        {
            "nome_specifico": sub.nome_specifico,
            "classe_astratta": sub.classe_astratta,
            "ingredienti": [
                {
                    "nome": ing.nome,
                    "quantita": ing.quantita,
                    "fase_utilizzo": getattr(ing, "fase_utilizzo", "Base"),
                }
                for ing in sub.ingredienti
            ],
        }
        for sub in draft.sotto_ricette
    ]

    # ==========================================
    # 3. ESTRAZIONE RADICE ONTOLOGICA
    # ==========================================
    prompt_estrazione_radice = f"""
    Analizza l'input originario dell'utente: '{state['input_utente']}' 
    e identifica il nome della ricetta base/madre di riferimento.

    REGOLE TASSONOMICHE RIGIDE:
    Devi estrarre la RADICE MADRE in questi due casi specifici:
    1. Modifiche dietetiche/salutistiche/cottura (es. 'Light', 'Vegana', 'Senza glutine', 'Al forno').
    2. Gusti, declinazioni o condimenti classici applicati a una base neutra (es. i gusti delle pizze, i sughi per la pasta, i tipi di risotto o torte).

    Se l'input è già una ricetta base senza specifiche aggiuntive (es. 'Caponata', 'Tiramisù', 'Pizza'), la radice madre sarà uguale all'input stesso.

    Esempi di conversione:
    'Pasta alla carbonara light' -> Radice: 'Pasta alla carbonara'
    'Tiramisù senza mascarpone' -> Radice: 'Tiramisù'
    'Pizza capricciosa' -> Radice: 'Pizza'
    'Pizza margherita' -> Radice: 'Pizza'
   
    """

    risultato_originale = llm_structured.invoke(
        [HumanMessage(content=prompt_estrazione_radice)]
    )
    topic_originale = risultato_originale.topic.capitalize()

    # Log di controllo per verificare l'estrazione sul terminale
    print(
        f" -> [DEBUG GEOMETRIA GRAFO]: Radice Madre: '{topic_originale}' | Output Finale: '{topic_finale}'"
    )

    # ==========================================
    # 4. SALVATAGGIO IN NEO4J
    # ==========================================
    try:
        print("\n===== DEBUG DATI KG =====")
        print(f"INGREDIENTI DIRETTI ({len(ingredienti_diretti)}):")
        print(ingredienti_diretti)
        print(f"\nSOTTO RICETTE ({len(sotto_ricette)}):")
        print(sotto_ricette)
        print("\nFONTE:")
        print(fonte)
        print("=========================\n")

        kg_client.salva_post(
            topic_originale=topic_originale,
            topic_finale=topic_finale,
            ingredienti_diretti=ingredienti_diretti,
            sotto_ricette=sotto_ricette,
            fonte=fonte,
        )

        print("[NEO4J] Salvataggio completato.")

    except Exception as e:
        print(f"[ERRORE NEO4J] Si è verificato un problema durante il salvataggio: {e}")

    return {}  # "indice_post_corrente":
    # state["indice_post_corrente"] + 1}


def aggiorna_topic_node(state: Blog_Cucina):
    """
    Usa un mini-LLM strutturato per estrarre in modo sicuro la variante scelta
    evitando la fragilità dello split testuale manuale.
    """
    print("\n--- [NODO: AGGIORNA TOPIC] ---")

    # 1. Recuperiamo il testo digitato dall'utente.
    # Se arriva dal tool chiedi_variante (interrupt nel ciclo di ricerca), lo troviamo nell'ultimo messaggio.
    # Se arriva dalla revisione finale, lo troviamo in human_feedback.
    ultimo_messaggio = state["messages"][-1] if state.get("messages") else None

    scelta_utente = state.get("human_feedback")

    if not scelta_utente and ultimo_messaggio:
        scelta_utente = ultimo_messaggio.content

    if not scelta_utente:
        print(" [AGGIORNA TOPIC] Nessun input utente trovato. Salto.")
        return {}

    # 2. Parsing sicuro con l'LLM Strutturato
    prompt_parsing = f"""L'utente ha selezionato una variante tra quelle proposte o ha indicato una nuova rotta. 
    Estrarre SOLO il nome pulito del piatto o della variante scelto da questo testo: '{scelta_utente}'"""

    risultato = llm_structured.invoke([HumanMessage(content=prompt_parsing)])
    nuovo_topic = risultato.topic.capitalize()

    print(
        f" [AGGIORNA TOPIC] Rilevata scelta utente. Nuovo topic impostato a: '{nuovo_topic}'"
    )
    return {"topic_corrente": nuovo_topic}
