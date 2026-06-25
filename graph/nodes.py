from langgraph.graph import END
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
    TopicPianificato,
    ValidationResult,
)
from graph.state import Blog_Cucina
from knowledge_graph.neo4j_manager import kg_client

# per il planner


llm_structured = llm.with_structured_output(TopicPianificato)


def planner_node(state: Blog_Cucina):

    print("\n--- [NODO 1: PLANNER (LLM)] ---")
    input = state["input_utente"]
    messaggi = state["messages"]
    feedback = state.get("human_feedback")

    lista = state.get("blacklist_topics") or []
    blacklist = ", ".join(lista) if lista else "Nessun topic scartato finora"

    if input == "PIANIFICAZIONE_AUTOMATICA":

        testo_prompt = f"""
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
    NON generare ancora il piano finale. Nella tua mente, elabora  SOLO 3 nuovi topic (variando le categorie in base alle tipolgie, se sono state pubblicati due primi e una secondo puoi generare un antipasto o un dessert, se una 
    tipolgia non ha publicata dai priorita a quest'ultima, se non sono stati pubblicati piati vegani  dai priorita ad un piatto vegano. L'obbiettivo è creare contenuti unici e rilevanti per i lettori del blog non annoiandoli).
   ⛔ prima di proporre qualsiasi piatto, devi assicurarti che NON sia nella blacklist: [{blacklist}].
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
    - ❌ SBAGLIATO (VIETATO): "primo a base di pesce"
    - ✅ CORRETTO: "Pollo al limone"
    - ✅ CORRETTO: "Verdure grigliate"
    - ✅ CORRETTO: "Filetto di manzo al pepe verde"
    - NON cercare dati online o in locale. Il tuo compito è solo pianificare.
    
    """

        if feedback and feedback.startswith("rigenera_con_blacklist:"):

            print(
                f"   [Planner] Rilevato feedback 'rigenera'. Topic scartati: {blacklist}"
            )

            testo_prompt += f"""
            
            --- FEEDBACK UTENTE (PIANO RIFIUTATO) ---
            ATTENZIONE: Il piano editoriale precedente è stato RIFIUTATO. 
            L'utente ha esplicitamente scartato i seguenti topic: {blacklist}.
            
            REGOLA ASSOLUTA: NON PROPORRE NESSUNO DI QUESTI TOPIC SCARTATI. 
            ⛔Devi generare 3 idee COMPLETAMENTE DIVERSE DA :{blacklist}.
            
            --- REGOLE DI WORKFLOW PER LA RIGENERAZIONE ---
            1. Nel tuo PRIMO utilizzo del `think_tool` in questo nuovo ciclo, DEVI dichiarare esplicitamente di aver ricevuto il feedback negativo e menzionare i topic scartati ({blacklist}) che eviterai.
            2. Chiama il tool `get_ultimi_post` per capire cosa è stato pubblicato di recente. (NON RICHIAMARLO PIù DI UNA VOLTA).
            3. Continua a usare il `think_tool` dopo ogni controllo con `controlla_storico_post`. 
            4. Usa la dicitura "STATO: CONTINUO" finché non trovi 3 nuove idee approvate dal tool.
            5. Usa la dicitura "STATO: FINITO" solo alla fine, quando hai i 3 topic definitivi.
            
            --- FORMATO DEI TOPIC (REGOLA FERREA) ---
            - Singola Preparazione: Il topic deve essere una ricetta reale, specifica e SINGOLA. 
            - È ASSOLUTAMENTE VIETATO combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
              - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
              - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
              - ❌ SBAGLIATO (VIETATO): "primo a base di pesce"
              - ✅ CORRETTO: "Pollo al limone"
              - ✅ CORRETTO: "Verdure grigliate"
              - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            
            [VINCOLO TASSATIVO]: NON cercare dati online o in locale. Il tuo compito è ESCLUSIVAMENTE pianificare internamente.
        
                        """

        if feedback and feedback.startswith("modifica:"):

            parti = feedback.split("|")
            istruzioni = parti[0].replace("modifica:", "").strip()
            piano_salvato = (
                parti[1].replace("piano:", "").strip() if len(parti) > 1 else ""
            )

            print(
                f"   [Planner] Rilevato 'modifica'. Istruzioni: {istruzioni} | Piano Vecchio: {piano_salvato} | Blacklist: {blacklist}"
            )

            testo_prompt += f"""
            --- FEEDBACK UTENTE (RICHIESTA DI MODIFICA) ---
            L'utente ha richiesto le seguenti MODIFICHE SPECIFICHE al piano: 
            "{istruzioni}"
            
            IL PIANO PRECEDENTE ERA COMPOSTO DA QUESTI 3 TOPIC: {piano_salvato}
            
            REGOLA MATEMATICA E LOGICA ASSOLUTA:
            1. Analizza la richiesta dell'utente: identifica QUALE topic deve essere eliminato e COSA l'utente vuole al suo posto (basandoti esclusivamente su "{istruzioni}").
            2 ⛔REGOLA FONDAMENTALE:NON DEVI PROPORRE TOPIC CONTENUTI IN {blacklist} in quanto sono stati scartati dall'utente.
            3. MANTIENI INTATTI E NON CANCELLARE gli altri topic del piano precedente che non sono stati nominati per l'eliminazione.
            4. Il NUOVO topic generato DEVE RISPETTARE le modifiche richieste dall'utente nella maniera più veritiera possibile.
            5. Alla fine del processo, il piano finale DEVE contenere rigorosamente: i topic vecchi mantenuti + il nuovo topic approvato (Totale esatto: 3 topic).
           
            --- REGOLE DI WORKFLOW PER LA MODIFICA ---
            1. Nel tuo PRIMO `think_tool`, dichiara: quali topic tieni, quale elimini, e QUANTI nuovi topic devi cercare (es. "Devo cercare 1 nuovo topic").
            2. Usa `controlla_storico_post` SOLO per verificare i NUOVI topic che stai introducendo. (Non ricontrollare i topic vecchi che hai deciso di tenere).
            3. 🚨 REGOLA DI STOP (IMPORTANTISSIMA): In ogni `think_tool`, DEVI FARE IL CONTEGGIO ESPLICITO. Scrivi letteralmente: "Topic mantenuti: X. Nuovi topic approvati: Y. Totale: Z". 
            4. Appena il tuo Totale Z arriva a 3, DEVI FERMARTI IMMEDIATAMENTE e chiudere il messaggio con "STATO: FINITO". È severamente vietato cercare un quarto topic.
            
            
                    ## Formato dei topic (REGOLA FERREA)
            - **Singola Preparazione:** Il topic deve essere una ricetta reale, specifica e SINGOLA. 
            - È ASSOLUTAMENTE VIETATO combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
            - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
            - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
            - ❌ SBAGLIATO (VIETATO): "primo a base di pesce"
            - ✅ CORRETTO: "Pollo al limone"
            - ✅ CORRETTO: "Verdure grigliate"
            - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            - NON cercare dati online o in locale. Il tuo compito è solo pianificare.
            
            [VINCOLO TASSATIVO]: NON cercare dati online o in locale. Il tuo compito è ESCLUSIVAMENTE pianificare internamente.
            """

        prompt = SystemMessage(content=testo_prompt)

    else:
        testo_prompt = """
    Sei il Direttore Editoriale di un blog di cucina. Analizza la direttiva dell'utente, assicurandoti che non ci siano duplicati.

    REGOLE DEL FLUSSO AUTONOMO E RISOLUZIONE CONFLITTI (ReAct):
     FASE 1. VERIFICA INIZIALE: Usa SEMPRE `controlla_storico_post` per verificare se la direttiva dell'utente è già stata pubblicata.
       APPENA RICEVI LA RISPOSTA DEL TOOL, prima di fare qualsiasi altra cosa o chiamare altri tool di ricerca, DEVI chiamare il `think_tool`.
        
    
    
     FASE 2. PROCEDURA Di RAGIONAMENTO  (SEGUI ALLA LETTERA):
       se il database risponde "OK" la direttiva dell'utente non è ancora stata pubblicata. Non hai bisogno di proporre varianti. Vai direttamente alla FASE 3
       Se il database risponde "BLOCCATO",  direttiva dell'utente è già stata pubblicata. NON generare idee a caso. Esegui questa sequenza esatta SOLO SE il database risponde "BLOCCATO":
       - AZIONE 1: Chiama immediatamente il tool `get_ingredienti` passando il nome del piatto bloccato.
       - AZIONE 2: Attendi i risultati da Neo4j (gli ingredienti).
       - AZIONE 3: Chiama il tuo `think_tool` per ragionare sugli ingredienti estratti e scegli 2 opzioni.
            - Opzione A (Variante): Se il piatto è facilmente personalizzabile nei gusti , proponi una variante .
            - Opzione B (Ricetta Simile): Se il piatto è una preparazione tradizionale strutturata (es. Arancini, Carbonara, Lasagne), usa gli ingredienti per trovare un piatto simile ma concettualmente diverso (es. "Supplì", "Gricia", "Cannelloni").

        - AZIONE 4: Dopo aver ragionato ed elaborato la nuova idea, verifica la TUA NUOVA proposta chiamando di nuovo `controlla_storico_post`.
        - Ripeti il ciclo se la nuova idea è ancora bloccata. SE una tua idea riceve "OK" vai alla FASE 3
    

     FASE 3. CONCLUSIONE:
       - Solo quando avrai ottenuto un "OK" , il topic che hai generato andrà bene e  potrai concludere.
       - Termina scrivendo: "STATO: FINITO".
       
        --- FORMATO DEI TOPIC (REGOLA FERREA) ---
            - Singola Preparazione: Il topic deve essere una ricetta reale, specifica e SINGOLA. 
            - È ASSOLUTAMENTE VIETATO combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
              - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
              - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
              - ❌ SBAGLIATO (VIETATO): "primo a base di pesce"
              - ✅ CORRETTO: "Pollo al limone"
              - ✅ CORRETTO: "Verdure grigliate"
              - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            
            [VINCOLO TASSATIVO]: NON cercare dati online o in locale. Il tuo compito è ESCLUSIVAMENTE proporre il topic in caso di duplicato.
    """

        # 2. GESTIONE DEL FEEDBACK: RIGENERA TOTALE CON BLACKLIST
        if feedback == "rigenera":
            print(f"   [Planner] Rilevato 'rigenera' singolo. Blacklist: {blacklist}")
            testo_prompt += f"""
            
            --- FEEDBACK UTENTE (PROPOSTA SINGOLA RIFIUTATA) ---
            ATTENZIONE: La tua idea precedente è stata BOCCIATA e si trova debtro la  {blacklist} . 
            
            REGOLA ASSOLUTA: NON PROPORRE NESSUNO DI QUESTI TOPIC SCARTATI NELLA TUA BLACKLIST perche l'utente li ha bocciati:
            [{blacklist}]
            
            Devi elaborare un'idea COMPLETAMENTE NUOVA e DIVERSA evitando i dati contenunti in:  {blacklist}.
            
            --- REGOLE DI WORKFLOW PER LA RIGENERAZIONE ---
            1. Nel tuo PRIMO utilizzo del `think_tool`, dichiara di aver ricevuto il feedback negativo e menziona la blacklist che eviterai a tutti i costi.
            2. Usa `controlla_storico_post` per verificare ESCLUSIVAMENTE 2la tua NUOVA idea.
            3. Se risulta "BLOCCATO", applica la normale procedura di risoluzione conflitti.
            4. 🚨 REGOLA DI STOP: Appena ricevi un "OK" definitivo, DEVI FERMARTI IMMEDIATAMENTE e chiudere il messaggio con "STATO: FINITO".
            
            
            
            ## Formato dei topic (REGOLA FERREA)
            - **Singola Preparazione:** Il topic deve essere una ricetta reale, specifica e SINGOLA. 
            - È ASSOLUTAMENTE VIETATO combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
            - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
            - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
            - ❌ SBAGLIATO (VIETATO): "primo a base di pesce"
            - ✅ CORRETTO: "Pollo al limone"
            - ✅ CORRETTO: "Verdure grigliate"
            - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            - NON cercare dati online o in locale. Il tuo compito è solo pianificare.
            non devi mai proporre nessuno di questi topic [{blacklist}]!!
            """

        # 3. GESTIONE DEL FEEDBACK: MODIFICA GUIDATA
        if feedback and feedback.startswith("modifica:"):

            istruzioni = feedback.replace("modifica:", "").strip()

            print(
                f"   [Planner] Rilevato 'modifica' singolo. Istruzioni: {istruzioni}. Blacklist: {blacklist}"
            )

            testo_prompt += f"""
            
            --- FEEDBACK UTENTE (RICHIESTA DI MODIFICA) ---
            L'utente ha richiesto le seguenti MODIFICHE SPECIFICHE alla tua singola proposta: 
            "{istruzioni}"
            
            
            
            REGOLA MATEMATICA E LOGICA ASSOLUTA:
            1. Analizza la richiesta dell'utente: identifica COSA l'utente vuole cambiare rispetto alla proposta scartata.
            2. Il NUOVO topic generato DEVE RISPETTARE ALLA LETTERA l'istruzione di modifica richiesta dall'utente.
            3. NON PROPORRE MAI i seguenti topic: {blacklist}.
            4. Alla fine del processo, devi avere rigorosamente 1 singolo topic approvato.
            
            --- REGOLE DI WORKFLOW PER LA MODIFICA ---
            1. Nel tuo PRIMO `think_tool`, dichiara: qual era il topic scartato e quale nuova strada hai deciso di intraprendere basandoti sulle istruzioni dell'utente.
            2. Usa `controlla_storico_post` SOLO per verificare il NUOVO topic che stai introducendo.
            3. Se il nuovo topic risulta "BLOCCATO", applica la normale procedura di risoluzione conflitti (usa `get_ingredienti` e poi ragiona su Variante o Ricetta Simile).
            4. 🚨 REGOLA DI STOP (IMPORTANTISSIMA): Appena ricevi un "OK" definitivo dal database per la tua nuova proposta, DEVI FERMARTI IMMEDIATAMENTE e chiudere il messaggio con "STATO: FINITO". È severamente vietato proporre idee aggiuntive o continuare a usare il think_tool.
                        
                                
            # Formato dei topic (REGOLA FERREA)
            - **Singola Preparazione:** Il topic deve essere una ricetta reale, specifica e SINGOLA. 
            - È ASSOLUTAMENTE VIETATO combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
            - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
            - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
            - ❌ SBAGLIATO (VIETATO): "primo a base di pesce"
            - ✅ CORRETTO: "Pollo al limone"
            - ✅ CORRETTO: "Verdure grigliate"
            - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            - NON cercare dati online o in locale. Il tuo compito è solo pianificare.          
            [VINCOLO TASSATIVO]: NON cercare dati online o in locale. Il tuo compito è ESCLUSIVAMENTE pianificare internamente.
            """

        prompt = SystemMessage(content=testo_prompt)

    # Ritorno da un Tool (il ragionamento è in corso)
    if messaggi and isinstance(messaggi[-1], ToolMessage):
        risposta_llm = llm_con_tools.invoke([prompt] + messaggi)
        return {"messages": [risposta_llm], "nodo_chiamante": "planner"}

    # Ritorno da Feedback Umano (I messaggi sono vuoti)
    elif feedback and not messaggi:
        print("   [Planner] Riavvio agente con innesco dinamico...")

        # in base al feedback
        if "rigenera" in feedback:

            # Distinguiamo tra pianificazione automatica e post singolo
            if input == "PIANIFICAZIONE_AUTOMATICA":
                idee = "3 idee COMPLETAMENTE DIVERSE"
                stop = "Appena hai 3 topic approvati"
            else:
                idee = "1 SOLA idea COMPLETAMENTE DIVERSA"
                stop = "Appena ricevi 1 solo OK definitivo"

            messaggio = (
                "L'utente ha rifiutato la tua ultima proposta.\n"
                "⚠️ LA TUA PRIMA AZIONE IN ASSOLUTO DEVE ESSERE CHIAMARE IL `think_tool`.\n"
                "Nel `think_tool` DEVI scrivere testualmente: 'Non proporrò i seguenti piatti "
                f"perché sono nella blacklist: [{blacklist}]'.\n"
                f"Solo dopo aver scritto questo, puoi iniziare a elaborare {idee}.\n"
                f"🚨 REGOLA DI STOP: {stop}, FERMATI IMMEDIATAMENTE con 'STATO: FINITO'."
            )

        else:  # caso modifica

            if input == "PIANIFICAZIONE_AUTOMATICA":
                idee = "1 NUOVO topic (mantenendo gli altri 2 del piano precedente)"
            else:
                idee = "1 SOLO nuovo topic"

            messaggio = (
                "L'utente ha richiesto una modifica specifica.\n"
                "⚠️ LA TUA PRIMA AZIONE IN ASSOLUTO DEVE ESSERE CHIAMARE IL `think_tool`.\n"
                f"Nel `think_tool` DEVI confermare di aver capito la modifica e che cercherai {idee}.\n"
                f"Blacklist da ignorare: [{blacklist}].\n"
                "🚨 REGOLA DI STOP: Appena ricevi 1 OK definitivo, FERMATI con 'STATO: FINITO'."
            )

        messaggi_da_inviare = [
            prompt,
            HumanMessage(content=messaggio),
        ]
        risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)

        return {"messages": [risposta_llm], "nodo_chiamante": "planner"}

    #  Primissimo avvio - Pianificazione Automatica
    elif input == "PIANIFICAZIONE_AUTOMATICA":
        messaggi_da_inviare = [
            prompt,
            HumanMessage(content="Inizia la ricerca..."),
        ]
        risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)
        return {"messages": [risposta_llm], "nodo_chiamante": "planner"}

    # Primissimo avvio - Post Singolo
    else:
        input_utente = state["input_utente"]

        # Estraiamo il topic principale SOLO al primo avvio

        risultato = llm_structured.invoke(
            [
                HumanMessage(
                    content=f"Estrai il piatto principale e la sua categoria da questa richiesta: '{input_utente}'. Lascia vuota la giustificazione."
                )
            ]
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
            "topic_corrente": topic_estratto,
            "nodo_chiamante": "planner",
        }


def human_review_planner(state: Blog_Cucina):
    print("\n--- [NODO: HUMAN REVIEW PLANNER] ---")

    feedback = interrupt({"msg": "In attesa di revisione umana"})
    print(f" Feedback ricevuto: {feedback}")

    piano_attuale = state.get("piano_editoriale")

    # Memoria persistente dei topic rifiutati nella sessione
    blacklist_attuale = list(state.get("blacklist_topics") or [])

    vecchio_feedback = state.get("human_feedback", "")

    # Se il turno prima era una modifica, scopriamo quale piatto ha eliminato l'LLM
    if (
        vecchio_feedback
        and "modifica:" in vecchio_feedback
        and "|piano:" in vecchio_feedback
        and piano_attuale
    ):
        vecchio_piano_str = vecchio_feedback.split("|piano:")[1]
        vecchi_topic = [t.strip() for t in vecchio_piano_str.split(",")]
        nuovi_topic = [post.topic for post in piano_attuale.sequenza_post]

        for t in vecchi_topic:
            # Se un piatto c'era prima, e ora non c'è più, va nella blacklist!
            if t not in nuovi_topic and t not in blacklist_attuale:
                blacklist_attuale.append(t)

    if feedback == "APPROVA":

        topic = piano_attuale.sequenza_post[0].topic

        print(" [NODO] Topic approvato! Avvio la pipeline di stesura.")

        messaggi_da_cancellare = [RemoveMessage(id=m.id) for m in state["messages"]]

        return Command(
            update={
                "topic_corrente": topic,
                "human_feedback": None,
                "blacklist_topics": blacklist_attuale,
                "messages": messaggi_da_cancellare,
            },
            goto="research",
        )

    elif feedback == "RIGENERA":

        topic_correnti = (
            [post.topic for post in piano_attuale.sequenza_post]
            if piano_attuale
            else []
        )

        nuova_blacklist = list(set(blacklist_attuale + topic_correnti))

        print(
            f" [Planner] Rigenerazione richiesta. "
            f"Aggiunti alla blacklist: {topic_correnti}"
        )

        messaggi_da_cancellare = [
            RemoveMessage(id=m.id) for m in state.get("messages", [])
        ]

        return Command(
            update={
                "human_feedback": "rigenera",
                "blacklist_topics": nuova_blacklist,
                "piano_editoriale": None,
                "messages": messaggi_da_cancellare,
            },
            goto="planner",
        )

    elif feedback.startswith("MODIFICA:"):

        istruzioni_modifica = feedback.replace("MODIFICA:", "").strip()

        # 1. Salviamo i topic ATTUALI in una stringa per passarli all'LLM.
        topic_correnti_str = (
            ", ".join([post.topic for post in piano_attuale.sequenza_post])
            if piano_attuale
            else ""
        )

        print(f" [Planner] Modifica richiesta. Inoltro le istruzioni all'Agente...")

        messaggi_da_cancellare = [
            RemoveMessage(id=m.id) for m in state.get("messages", [])
        ]

        return Command(
            update={
                # 2. FONDAMENTALE: Alleghiamo le istruzioni E i topic attuali
                "human_feedback": f"modifica:{istruzioni_modifica}|piano:{topic_correnti_str}",
                # 3. NON TOCCHIAMO LA BLACKLIST! Passiamo quella attuale intatta
                "blacklist_topics": blacklist_attuale,
                "piano_editoriale": None,
                "messages": messaggi_da_cancellare,
            },
            goto="planner",
        )


# variante
def human_review_variante(state: Blog_Cucina):

    print("\n--- [NODO: HUMAN REVIEW VARIANTE / SINGOLO] ---")

    feedback = interrupt(
        {"msg": "Proposta singola pronta. In attesa di approvazione umana..."}
    )

    print(f" [NODO] Istruzione ricevuta dal main: {feedback}")

    topic_proposto = state.get("topic_corrente")

    blacklist_attuale = list(state.get("blacklist_topics") or [])

    if feedback == "APPROVA":

        print(" [NODO] Topic approvato! Avvio la pipeline di stesura.")
        messaggi_da_cancellare = [RemoveMessage(id=m.id) for m in state["messages"]]

        return Command(
            update={
                "messages": messaggi_da_cancellare,
                "human_feedback": None,
                "blacklist_topics": blacklist_attuale,
            },
            goto="research",
        )

    elif feedback == "RIGENERA":

        print(" [NODO] Topic rifiutato. Avvio nuova generazione.")

        if (
            topic_proposto
            and topic_proposto != "None"
            and topic_proposto not in blacklist_attuale
        ):
            blacklist_attuale.append(topic_proposto)

        messaggi_da_cancellare = [
            RemoveMessage(id=m.id) for m in state.get("messages", [])
        ]

        return Command(
            update={
                "human_feedback": "rigenera",
                "blacklist_topics": blacklist_attuale,
                "topic_corrente": None,
                "messages": messaggi_da_cancellare,
            },
            goto="planner",
        )

    elif feedback.startswith("MODIFICA:"):

        istruzioni = feedback.replace("MODIFICA:", "").strip()

        print(" [NODO] Richiesta modifica ricevuta.")

        if (
            topic_proposto
            and topic_proposto != "None"
            and topic_proposto not in blacklist_attuale
        ):
            blacklist_attuale.append(topic_proposto)

        messaggi_da_cancellare = [
            RemoveMessage(id=m.id) for m in state.get("messages", [])
        ]

        return Command(
            update={
                "human_feedback": f"modifica:{istruzioni}",
                "blacklist_topics": blacklist_attuale,
                "topic_corrente": None,
                "messages": messaggi_da_cancellare,
            },
            goto="planner",
        )


def krag_research_node(state: Blog_Cucina):
    print("\n--- [NODO 2: RICERCA MCP (Agente Autonomo)] ---")
    topic = state["topic_corrente"]
    post_rifiutato = state.get("post_draft", "")
    is_rigenera = state.get("is_rigenera", False)
    reasoning_trace = state.get("reasoning_trace", [])
    riflessioni_research = [r for r in reasoning_trace if r.startswith("[RESEARCH]")]
    print(f"{topic}")
    messaggi = state.get("messages", [])

    testo_base = f"""
    Sei un Agente Investigatore esperto, specializzato in recupero dati culinari. Il tuo obiettivo è raccogliere DATI per la ricetta completa per il topic: '{topic}' e TUTTE le sue eventuali sottoricette.
    seguendo l'algoritmo. Solo quando avrai completato l'albero delle sottoricette di TUTTE le Ricette Madri potrai dichiarare "STATO: FINITO".
    ---
    ###  REGOLE GENERALI (LEGGI CON ATTENZIONE)
    
            
        ### CLASSIFICAZIONE PRELIMINARE DEL TOPIC (FONDAMENTALE)
            Prima di applicare l'algoritmo, classifica mentalmente il topic in uno di questi tipi:

            **TIPO A – RICETTA COMPLESSA CON SOTTORICETTE**  
            Preparazioni che richiedono altre preparazioni  (es. lasagne,setteveli,insalata russa, ecc...).  
            → Applica l'algoritmo ricorsivo completo.

            **TIPO B – RICETTA SEMPLICE / ASSEMBLAGGIO**  
            Piatti dove gli ingredienti vengono usati direttamente, al massimo con semplici operazioni di taglio, schiacciatura o miscelatura (es. bruschette, insalate, toast, carpacci, macedonie).  
            → Raccogli la ricetta madre e FERMATI. **Non cercare sottoricette**. Se trovi più versioni, raccoglile come documenti separati ma non cercare sottoricette per nessuna.
            

        1. **PENSIERO OBBLIGATORIO**: Prima di chiamare QUALSIASI tool di ricerca("esegui_ricerca_web" e "cerca_ricetta_nel_db"), DEVI chiamare `think_tool` spiegando:
            - In quale FASE ti trovi
            - Cosa stai per fare e perché
            - Concludi con "STATO: CONTINUO" (o "STATO: FINITO" se hai completato TUTTO)

        2. **QUERY ESPANSE (get_ingredienti)**:
            - Per OGNI nuovo elemento che cerchi (Ricetta Madre o sottoricetta), DEVI chiamare `get_ingredienti` UNA VOLTA per ottenere una query espansa.
            - Se stai ritentando la ricerca dello STESSO elemento (es. dopo fallimento), NON chiamare di nuovo `get_ingredienti` – riutilizza la query che hai già.
            - Usa SEMPRE la query espansa per `cerca_ricetta_nel_db`.
        

  
    ### L'ALGORITMO DI RICERCA

    Per OGNI elemento che devi cercare (partendo da '{topic}', poi ogni sottoricetta), esegui questa sequenza:
  
    ▶ **FASE 1: RICERCA LOCALE (DB FIRST)**

        1. **OTTIENI QUERY ESPANSA**: 
        - Chiama `get_ingredienti` per l'elemento corrente (se non l'hai già fatto).
        - Usa il risultato per costruire una query dettagliata.

        2. **CERCA NEL DB**: 
        - Usa `cerca_ricetta_nel_db` con la query espansa.

        3. **VALUTA IL RISULTATO**:
        - ✅ Se TROVATA una ricetta COMPLETA con ingredienti e procedimento chiari e coerente al topic richiesto dall'utente:
            - Memorizza i dati.
            - **VIETATO** usare il web per questa ricetta.
            - Vai alla **FASE 2**.
        - ❌ Se NON TROVATA, INCOMPLETA o incoerente al topic richiesto:
            - chiama il think tool per analizzare il fallimento
            - La ricerca locale è fallita.
            - Vai al punto 4.
            
        4. **RICERCA WEB** (solo se il DB ha fallito):
        - Usa `esegui_ricerca_web` per cercare l'elemento corrente.
        - Memorizza i dati e vai alla **FASE 2**.
    
    ▶ **FASE 2: IL LOOP SOTTORICETTE**

        Per OGNI documento WEB o DB della ricetta appena acquisita:

        1. Leggi attentamente ingredienti e procedimento.
        2. Cerca eventuali SOTTORICETTE applicando questi criteri:
        - **CRITERIO ESPLICITO**: Se un ingrediente dice "(vedi preparazione base)" → è una sottoricetta.
        - **CRITERIO DEDUTTIVO**: Se la ricetta richiede una preparazione complessa (besciamella, ragù, crema, pasta biscotto, glassa, pastella, ecc.) nei suoi ingredienti o passaggi → è una sottoricetta.

        3. **Esito dell'analisi**:
        - **NESSUNA SOTTORICETTA**: 
            - Usa `think_tool` e scrivi "STATO: FINITO" per questa Ricetta Madre.
            - Se ci sono altre Ricette Madri, passaci.
        - **SOTTORICETTE TROVATE**: 
            - Astrai il loro VERO NOME.
            - Per OGNUNA, esegui la **FASE 3**.
    
    ▶ **FASE 3: LOOP SOTTORICETTE (eseguito per OGNI sottoricetta trovata)**

    Per OGNI sottoricetta (es. "Maionese", "Besciamella", "Pasta biscotto"):

    **CHECK 1: RICERCA LOCALE (DB) — PRIORITÀ ASSOLUTA**

        1. **OTTIENI QUERY ESPANSA**:
        - Se è la prima volta che cerchi questa sottoricetta, chiama `get_ingredienti` per essa.
        - Usa la query espansa per `cerca_ricetta_nel_db`.

        2. **CERCA NEL DB**:
        - Usa `cerca_ricetta_nel_db` con la query espansa.

        3. **VALUTA**:
        - ✅ Se TROVATA una ricetta COMPLETA (ingredienti + procedimento):
            - La sottoricetta è RISOLTA. Memorizza i dati.
            - **PASSA ALLA PROSSIMA SOTTORICETTA**.
        - ❌ Se NON TROVATA o INCOMPLETA:
            - Procedi al **CHECK 2**.


    **CHECK 2: CORTOCIRCUITO (MEMORIA DELLA RICETTA MADRE)**

        1. Usa `think_tool` per analizzare la Ricetta Madre che hai in memoria.
        2. Scrivi la riflessione in questo formato:

        --- ANALISI ---
        [Nome sottoricetta]: è già descritta nei passaggi della Ricetta Madre?

        3. **Valuta**:
        - ✅ Se SÌ: "CORTOCIRCUITO: [Nome] è già inclusa nei passaggi X,Y,Z. STATO: FINITO"
            - La sottoricetta è RISOLTA. NON cercare altro.
            - **PASSA ALLA PROSSIMA SOTTORICETTA**.
        - ❌ Se NO: "CORTOCIRCUITO FALLITO: [Nome] non è descritta nella Ricetta Madre. STATO: CONTINUO"
            - Procedi al **CHECK 3**.

    **CHECK 3: RICERCA WEB (ULTIMA SPIAGGIA)**

        1. Solo se il **CHECK 2** ha dato esito negativo, usa `esegui_ricerca_web` per cercare la sottoricetta.

        2. **Valuta**:
        - ✅ Se TROVATA: Memorizza, la sottoricetta è RISOLTA. **PASSA ALLA PROSSIMA**.
        - ❌ Se NON TROVATA: Dichiara "FALLIMENTO: [Nome] non trovata né in DB né in Web. ABBANDONO il ramo."
            - **PASSA ALLA PROSSIMA** (NON ripetere).


    ▶ **TERMINAZIONE**

    - Dopo aver risolto o abbandonato TUTTE le sottoricette di una Ricetta Madre, la Ricetta Madre è COMPLETA.
    - Se ci sono altre Ricette Madri, processale una alla volta con lo stesso algoritmo.
    - Solo quando TUTTE le Ricette Madri e TUTTE le loro sottoricette sono risolte, usa `think_tool` e scrivi "STATO: FINITO".

    ### STATO ATTUALE

    Sei pronto ad agire. Inizia invocando `think_tool` dichiarando l'avvio della MACRO-FASE 1 per '{topic}'.
    """

    riflessioni_research = [r for r in reasoning_trace if r.startswith("[RESEARCH]")]

    # contesto rigenera
    print(f"{is_rigenera}")
    if is_rigenera:
        print(f"post rifiutao: {post_rifiutato}")
        if post_rifiutato:
            contesto = f"""
            ###  RIGENERAZIONE POST ###
            Il post precedente su '{topic}' è stato RIFIUTATO dall'utente.
            Bozza rifiutata (inizio): {post_rifiutato}...
            Cerca NUOVE fonti o dettagli che possano rendere la nuova versione più completa, affidabile o interessante.
            Non ripetere esattamente la bozza precedente.
            """
            testo_base = contesto + "\n" + testo_base

    if messaggi and riflessioni_research:

        if is_rigenera:
            # Modalità  rigenerazione mostriamo tutto il ragionamento
            riepilogo_trace = "\n".join(f"- {r}" for r in reasoning_trace)
        else:
            # Modalità normale mostriamo solo le riflessioni del research
            riepilogo_trace = "\n".join(
                f"- {r.replace('[RESEARCH] ', '')}" for r in riflessioni_research
            )

        contesto_trace = f"""
        ### RIEPILOGO DEL TUO RAGIONAMENTO FINORA
        Usale per non ripetere azioni già fatte e decidere il passo successivo:
        {riepilogo_trace}
        """
        testo_base = testo_base + contesto_trace

    prompt_finale = SystemMessage(content=testo_base)

    if not messaggi:
        # Prima entrata o inizio rigenerazione
        messaggi_da_inviare = [
            prompt_finale,
            HumanMessage(content=f"Inizia la ricerca per il topic: {topic}"),
        ]
    else:
        # Ciclo ReAct in corso
        messaggi_da_inviare = [prompt_finale] + messaggi

    risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)

    return {
        "messages": [risposta_llm],
        "nodo_chiamante": "research",
    }


def validator_node(state: Blog_Cucina):
    print("\n--- [NODO 3: VALIDATORE (Fact-Checking Incrociato)] ---")
    topic = state["topic_corrente"]
    ragionamento = state.get("reasoning_trace")
    riflessioni_research = [r for r in ragionamento if r.startswith("[RESEARCH]")]
    dati_db_locale = state.get("rag_documents", [])
    dati_web_grezzi = state.get("web_documents", [])
    messaggi = state.get("messages", [])
    totale_doc = len(dati_db_locale) + len(dati_web_grezzi)

    testo_db = "NESSUNA RICETTA TROVATA NEL DB LOCALE"
    if dati_db_locale:
        testo_db = "".join(
            f"\n=== DB_DOC_{idx} ===\n{doc}\n" for idx, doc in enumerate(dati_db_locale)
        )

    testo_web = "NESSUN DATO COLLATERALE DAL WEB"
    if dati_web_grezzi:
        testo_web = "".join(
            f"\n=== WEB_DOC_{idx} ===\n{doc}\n"
            for idx, doc in enumerate(dati_web_grezzi)
        )

    # ── PRIMA PASSATA: messaggi vuoti → manda al think_tool ──
    if not messaggi:

        prompt_riflessione = f"""
        Sei un Validatore Supremo esperto di ricettari e un rigoroso risolutore di dipendenze. 
        Ti sono stati forniti ESATTAMENTE {totale_doc} documenti in totale (tra DB locale e WEB).

        🚨 ALLERTA ROSSA: QUESTA È L'UNICA INTERAZIONE CHE AVRAI. NON CI SARÀ NESSUN "PROSSIMO TURNO".
        DEVI ESEGUIRE TUTTE E 3 LE FASI ORA, ALL'INTERNO DI QUESTA SINGOLA CHIAMATA AL TOOL.
        USARE "STATO: CONTINUO" È UN ERRORE FATALE CHE COMPROMETTERÀ IL SISTEMA.

        Compila i 3 campi del `think_tool` SEGUENDO TASSATIVAMENTE QUESTE ISTRUZIONI:

        ▶ NEL CAMPO 'analisi_contesto' (ESEGUI FASE 1 e FASE 2 INSIEME ADESSO):
            - Confronta TUTTI i documenti che trattano '{topic}'.
           - Eleggi il MIGLIORE come "Ricetta Madre" (assegnali SCORE 1) in base al punteggio o all'autorevolezza.
           - Assegna SCORE 0 a tutti gli altri documenti che parlano di '{topic}' (sono duplicati inferiori).
           - Analizzando i tuoi ragionamenti precedenti {riflessioni_research} individua le sottoricette che sono strettamente necessarie per realizzarla (non citare gli ingredienti).

        ▶ NEL CAMPO 'valutazione_opzioni' (ESEGUI FASE 3 ADESSO):
        - Prendi i documenti RIMASTI (quelli non ancora eletti o scartati).  - 
           - SE il documento è una sottoricetta della ricetta madre: ASSEGNA Score 1 al migliore e scarta gli altri assegandoli Score 0(sono duplicati inferiori).
           - SE il documento è irrilevante,fuori tema o non serve: assegna SCORE 0.

        ▶ NEL CAMPO 'decisione_finale' (VERDETTO FINALE):
        - Scrivi la LISTA FISICA ED ESATTA delle tue valutazioni.
        - L'elenco DEVE contenere ESATTAMENTE {totale_doc} righe (una per ogni documento che ti ho fornito).
        - Formato obbligatorio per ogni riga:
          ID_DOC [TITOLO]: Score X - Motivo: ...
        - Dopo aver scritto l'elenco completo, DEVI CONCLUDERE TASSATIVAMENTE CON: "STATO: FINITO".

        === DOCUMENTI DA VALUTARE ===
        [DB LOCALE]
        {testo_db}

        [WEB]
        {testo_web}
        """
        messaggio = [
            SystemMessage(content=prompt_riflessione),
            HumanMessage(content=f"Analizza i documenti per '{topic}'."),
        ]
        risposta = llm_con_tools.invoke(messaggio)

        return {"messages": [risposta], "nodo_chiamante": "validator"}

    # ── SECONDA PASSATA: leggi la riflessione dai messaggi e dai il verdetto ──
    # Il tool_node ha già stampato il ragionamento, ci basta il contenuto
    riflessione_testo = ""
    for msg in reversed(messaggi):
        if isinstance(msg, ToolMessage) and msg.name == "think_tool":
            riflessione_testo = msg.content.replace(
                "Riflessione registrata con successo: ", ""
            ).strip()
            break

    prompt_verdetto = f"""
        Sulla base esclusiva di questa tua analisi logica appena effettuata:
        {riflessione_testo}

        Produci il verdetto strutturato finale per il topic '{topic}'.
        
        ATTENZIONE (CRITICO): 
        1. Devi estrarre e mappare TUTTI i documenti a cui hai assegnato SCORE 1 nella tua analisi.
        2. Usa gli STESSI IDENTICI ID_DOC (i numeri esatti) che hai scritto nell'analisi. È severamente vietato inventare ID non presenti nel testo.
        3. Puoi (e DEVI) approvare UNO O PIÙ documenti (sia DB che Web) se uno rappresenta la Ricetta Madre e gli altri sono le Sottoricette necessarie.
        
        Regole per i flag:
        - is_valid: True SE E SOLO SE i dati totali approvati (Score 1) sono sufficienti e coerenti per scrivere il post. Se nessun documento ha score 1, imposta a False.
        - usa_db_locale: True se almeno UN documento proveniente dal DB locale è stato approvato (Score 1).
        """
    esito = llm.with_structured_output(ValidationResult).invoke(
        [HumanMessage(content=prompt_verdetto)]
    )

    print(f"\n Esito Validazione (is_valid): {esito.is_valid}")
    print(f" Motivazione Generale: {esito.motivazione_qualita}")

    # re-ranking e pruning dati db

    dati_db_filtrati = []

    for d in esito.ranking_db:
        print(d.id, d.score, d.motivo)

    if dati_db_locale and esito.ranking_db:
        db_ordinato = sorted(esito.ranking_db, key=lambda x: x.score, reverse=True)

        print(f"{db_ordinato}")

        for d in db_ordinato:

            if d.score == 1:

                dati_db_filtrati.append(dati_db_locale[d.id])

        print(f"{dati_db_filtrati}")

    # re-ranking e pruning dati web

    dati_web_filtrati = []

    for d in esito.ranking_web:
        print(d.id, d.score, d.motivo)

    if dati_web_grezzi and esito.ranking_web:

        web_ordinato = sorted(esito.ranking_web, key=lambda x: x.score, reverse=True)

        for d in web_ordinato:

            if d.score == 1:

                dati_web_filtrati.append(dati_web_grezzi[d.id])

    messaggi_da_cancellare = [RemoveMessage(id=m.id) for m in messaggi]

    return {
        "messages": messaggi_da_cancellare,
        "is_valid": esito.is_valid,
        "approved_web_documents": dati_web_filtrati,
        "approved_db_documents": dati_db_filtrati,
    }


# MARKDOWN PER LA BOZZA
def genera_markdown_bozza(draft) -> str:
    """Trasforma il Pydantic RecipeDraft in un post Markdown pulito e scansionabile."""

    md = f"# {draft.titolo}\n\n"
    md += f"## Introduzione\n{draft.introduzione}\n\n"

    md += "## Ingredienti Principali\n"
    if draft.ingredienti_diretti:
        for ing in draft.ingredienti_diretti:
            md += f"- **{ing.nome}**: {ing.quantita}\n"
    else:
        md += "- *Nessun ingrediente diretto aggiuntivo.*\n"

    if draft.sotto_ricette:
        md += "\n## Sottoricette Necessarie\n"
        for sub in draft.sotto_ricette:
            # [MODIFICA QUI] Stampa SOLO la classe astratta pulita (es. "Maionese" invece di "Maionese Fatta in Casa")
            md += f"\n### {sub.classe_astratta}\n"
            for ing in sub.ingredienti:
                md += f"- **{ing.nome}**: {ing.quantita}\n"

    md += "\n## Preparazione Passo-Passo\n"
    if draft.preparazione:
        for i, step in enumerate(draft.preparazione, start=1):
            md += f"{i}. {step}\n"
    else:
        md += "1. *Nessun passaggio specificato.*\n"

    # [MODIFICA QUI] Cicla su tutte le fonti raccolte e crea un elenco puntato
    md += f"\n---\n**Fonti Utilizzate**:\n"
    if hasattr(draft, "fonti") and draft.fonti:
        for f in draft.fonti:
            md += f"- {f}\n"
    else:
        md += "- *Fonte sconosciuta*\n"

    return md


# writer node: sintetizza le informazioni approvate e scrivi la bozza del post in markdown, con attenzione alla distinzione tra ingredienti diretti e sotto-ricette, e alla gerarchia degli ingredienti. Applica eventuali feedback umani ricevuti per correggere o migliorare la bozza prima di generare il markdown finale.


def writer_node(state):
    print("\n--- [NODO 4: WRITER (Sintesi, Grounding e Coesione)] ---")
    topic = state["topic_corrente"]
    dati_db_locale = state.get("approved_db_documents", [])
    dati_web = state.get("approved_web_documents", [])
    feedback = state.get("human_feedback", "")
    post_precedente = state.get("post_draft", "")

    lista_tracce = state.get("reasoning_trace", [])
    traces_formattate = (
        "\n".join([f"- {t}" for t in lista_tracce])
        if lista_tracce
        else "Nessuna traccia di ragionamento precedente registrata."
    )

    testi_approvati = "\n\n".join(dati_db_locale + dati_web)
    if not testi_approvati.strip():
        testi_approvati = "ERRORE: Nessun documento approvato dal Validatore."

    try:
        contesto_kg = kg_client.get_contesto_editoriale(topic)

        style = contesto_kg.get("style", {})
        claim_correlati = contesto_kg.get("claim_correlati", [])
        topic_correlati = contesto_kg.get("topic_correlati", [])

        # Blocco stile
        if style:
            blocco_stile = (
                f"Tono: {style.get('tono', 'n/d')}\n"
                f"Registro: {style.get('registro', 'n/d')}\n"
                f"Lunghezza target: {style.get('lunghezza', 'n/d')} parole\n"
                f"Audience: {style.get('audience', 'n/d')}\n"
                f"Note stilistiche: {style.get('note', '')}"
            )
        else:
            blocco_stile = (
                "Nessuna linea guida stilistica disponibile (primo post del blog)."
            )

        # Blocco claim dei post precedenti
        if claim_correlati:
            righe_claim = []
            for p in claim_correlati:
                claims_str = " | ".join(p["claims"][:3])
                righe_claim.append(f"  • [{p['topic']}]: {claims_str}")
            blocco_claim = "\n".join(righe_claim)
        else:
            blocco_claim = (
                "Nessun post precedente nel Knowledge Graph (primo post del blog)."
            )

        # Blocco topic correlati per ingredienti condivisi
        if topic_correlati:
            righe_correlati = []
            for c in topic_correlati:
                ing_str = ", ".join(c["ingredienti_comuni"])
                righe_correlati.append(
                    f"  • '{c['topic']}' (ingredienti in comune: {ing_str})"
                )
            blocco_correlati = "\n".join(righe_correlati)
        else:
            blocco_correlati = "Nessun topic correlato trovato nel Knowledge Graph."

        print(
            f"[WRITER] KG: stile={'ok' if style else 'vuoto'} | "
            f"claim_post={len(claim_correlati)} | correlati={len(topic_correlati)}"
        )

    except Exception as e:
        print(f"[WRITER] Errore recupero KG: {e}. Procedo senza contesto editoriale.")
        blocco_stile = "Contesto stilistico non disponibile."
        blocco_claim = "Claim precedenti non disponibili."
        blocco_correlati = "Topic correlati non disponibili."

    # ── Blocco modifica (solo se l'utente ha richiesto correzioni) ────────────
    if feedback:
        blocco_modifica = f"""
        ========================================================================
        MODALITÀ MODIFICA — ISTRUZIONI PRIORITARIE
        ========================================================================
        Il post seguente è già stato generato ma il redattore ha richiesto modifiche.
        Applica ESCLUSIVAMENTE le modifiche indicate, mantenendo intatte le parti non menzionate.
        Rispetta comunque le linee guida stilistiche del blog indicate sopra.

        MODIFICHE RICHIESTE:
        {feedback}

        POST PRECEDENTE DA MODIFICARE:
        {post_precedente}
    """
    else:
        blocco_modifica = ""

    # ── Prompt finale ─────────────────────────────────────────────────────────
    prompt = f"""
        Sei il Redattore Editoriale senior di un blog di cucina siciliana.
        Il tuo compito è scrivere la bozza finale strutturata per il post su: '{topic}'.

        Ti vengono forniti tre asset che costituiscono il tuo perimetro di lavoro:
        1. CONTESTO EDITORIALE DAL KNOWLEDGE GRAPH: lo stile del blog, i claim già
        pubblicati e i topic correlati — per garantire coerenza e creare rimandi.
        2. TESTI APPROVATI: i documenti con ricetta madre e sottoricette — la tua
        unica fonte di verità per ingredienti, dosi e procedimenti.
        3. TRACCE DI RAGIONAMENTO: il log dell'agente che ha scomposto il piatto
        e risolto le dipendenze — ti dice cosa è una sottoricetta autonoma.

        ========================================================================
        CONTESTO EDITORIALE DAL KNOWLEDGE GRAPH
        ========================================================================

        [STILE DEL BLOG — rispetta queste linee guida in ogni scelta redazionale]
        {blocco_stile}

        [CLAIM GIÀ PUBBLICATI — richiamali esplicitamente se pertinenti, evita di ripeterli]
        {blocco_claim}

        [TOPIC CON INGREDIENTI IN COMUNE — puoi creare rimandi a questi post correlati]
        {blocco_correlati}

        ========================================================================
        CRITERI DI COMPILAZIONE TASSATIVI
        ========================================================================

        1. COERENZA EDITORIALE (usa il contesto KG):
        - Se un claim di un post precedente è pertinente al topic corrente, citalo
            esplicitamente nell'introduzione o nei passaggi (es. "Come abbiamo visto
            nel post sulla Besciamella..."). Non inventare rimandi: usa solo quelli
            che trovi nel blocco CLAIM GIÀ PUBBLICATI.
        - Se esiste un topic correlato (ingredienti in comune), puoi aggiungere
            un breve rimando alla fine dell'introduzione (es. "Se ami questo piatto,
            potresti trovare interessante anche la nostra ricetta di X").
        - Rispetta SEMPRE il tono, il registro e la lunghezza target indicati
            nelle linee guida stilistiche.

        2. RIGORE STRUTTURALE (coesione ricetta → sottoricette):
        - Il tuo output deve rispecchiare fedelmente l'albero delle dipendenze
            stabilito nelle tracce di ragionamento.
        - Se il ragionamento indica che un ingrediente (es. Besciamella, Ragù)
            è una SOTTORICETTA autonoma, mappala interamente in 'sotto_ricette'
            estraendo i suoi ingredienti dal testo della fonte.
        - È VIETATO lasciare una sottoricetta complessa come stringa piatta
            negli ingredienti diretti della ricetta madre.

        3. ANCHORING METRICO-TESTUALE:
        - Estrai dosi, pesi, ingredienti e passaggi ESCLUSIVAMENTE dai testi
            approvati forniti. Non inventare ingredienti né arrotondare le dosi.
        - Se un ingrediente manca di quantità nel testo, scrivi "q.b." o
            "quantità non specificata".

        4. REGOLA ANTI-SOPRASTRUTTURE:
        - Non creare oggetti SottoRicetta per ingredienti pronti o topping che
            non subiscono trasformazione (es. parmigiano da spolverare, prosciutto
            pronto): questi vanno negli ingredienti diretti.

        5. DETTAGLIO DELLA PREPARAZIONE:
        - Elenca i passaggi in modo esteso, analitico e sequenziale.
        - Ogni passaggio è una stringa chiara e indipendente nella lista.
        - Non riassumere procedimenti complessi in un solo paragrafo.

        ========================================================================
        FONTI E RAGIONAMENTI
        ========================================================================

        [TRACCE DI RAGIONAMENTO (LOG DEL GRAPH STATE)]
        {traces_formattate}

        [TESTI DELLE FONTI APPROVATE (RICETTA MADRE + SOTTORICETTE)]
        {testi_approvati}

        {blocco_modifica}
        """

    llm_writer = llm.with_structured_output(RecipeDraft)
    draft = llm_writer.invoke([HumanMessage(content=prompt)])

    # ── Middleware: deduplicazione sottoricette dagli ingredienti diretti ─────
    nomi_sottoricette = set()
    for sub in draft.sotto_ricette:
        nomi_sottoricette.add(sub.classe_astratta.lower())
        nomi_sottoricette.add(sub.nome_specifico.lower())

    ingredienti_diretti_puliti = []
    for ing in draft.ingredienti_diretti:
        if not any(nome_sub in ing.nome.lower() for nome_sub in nomi_sottoricette):
            ingredienti_diretti_puliti.append(ing)

    draft.ingredienti_diretti = ingredienti_diretti_puliti

    print("\n===== DEBUG WRITER =====")
    print(draft.model_dump())
    print("========================\n")

    markdown_finale = genera_markdown_bozza(draft)

    return {
        "recipe_draft": draft,
        "post_draft": markdown_finale,
        "human_feedback": None,
    }


def human_review_node(state: Blog_Cucina):
    print("\n--- [NODO 5: HUMAN-IN-THE-LOOP (Approvazione)] ---")
    bozza = state.get("post_draft", "")

    print("\n================ BOZZA DEL POST ================\n")
    print(bozza)
    print("\n================================================\n")

    feedback = interrupt({"msg": "In attesa di revisione umana"})

    if feedback == "APPROVA":

        print(" [NODO] Topic approvato! Avvio la pipeline di stesura.")

        return Command(goto="kg_update")

    elif feedback == "RIGENERA":
        print("[HITL] Rigenerazione richiesta. Torno al research.")
        messaggi_da_cancellare = [
            RemoveMessage(id=m.id) for m in state.get("messages", [])
        ]
        return Command(
            update={
                "messages": messaggi_da_cancellare,
                "is_rigenera": True,
                "human_feedback": "",
                "rag_documents": [None],  # grazie a replace_or_add
                "web_documents": [None],
                "approved_db_documents": [],
                "approved_web_documents": [],
            },
            goto="research",
        )

    elif feedback.startswith("MODIFICA:"):
        testo_feedback = feedback.replace("MODIFICA:", "").strip()
        print(f"[HITL] Modifica richiesta: '{testo_feedback}'")
        return Command(update={"human_feedback": testo_feedback}, goto="writer")

    elif feedback == "SCARTA":
        print("[HITL]  Post scartato dall'utente.")
        return Command(
            update={
                "human_feedback": "",
                "post_draft": "",
            },
            goto=END,
        )


def kg_update_node(state: Blog_Cucina):
    print("\n--- [NODO 6: KG UPDATE (Aggiornamento Memoria)] ---")
    indice_corrente = state.get("indice_post_corrente", 0)
    draft = state.get("recipe_draft")
    if not draft:
        print("[ERRORE] Nessun recipe_draft trovato nello stato.")
        return {}

    topic_finale = state["topic_corrente"]
    fonte = draft.fonti

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
            testo_post=state.get("post_draft", ""),
            llm=llm,
        )

        print("[NEO4J] Salvataggio completato.")

    except Exception as e:
        print(f"[ERRORE NEO4J] Si è verificato un problema durante il salvataggio: {e}")

    return {
        "indice_post_corrente": indice_corrente + 1,
        "approved_db_documents": [],
        "approved_web_documents": [],
        "rag_documents": [None],
        "web_documents": [None],
        "is_valid": None,
        "human_feedback": "",
        "is_rigenera": False,
    }
