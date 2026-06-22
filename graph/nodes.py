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
    1. VERIFICA INIZIALE: Usa SEMPRE `controlla_storico_post` per verificare se la direttiva dell'utente è già stata pubblicata.
       APPENA RICEVI LA RISPOSTA DEL TOOL, prima di fare qualsiasi altra cosa o chiamare altri tool di ricerca, DEVI chiamare il `think_tool`.
    
    2. PROCEDURA DI BLOCCO E RAGIONAMENTO RAG (SEGUI ALLA LETTERA):
       Se il database risponde "BLOCCATO", NON generare idee a caso. Esegui questa sequenza esatta:
       - AZIONE 1: Chiama immediatamente il tool `ottieni_ingredienti_ricetta` passando il nome del piatto bloccato.
       - AZIONE 2: Attendi i risultati da Neo4j (gli ingredienti).
       - AZIONE 3: Chiama il tuo `think_tool` per ragionare sugli ingredienti estratti.
       
       NEL THINK_TOOL, SCEGLI LA STRATEGIA:
       - Opzione A (Variante): Se il piatto è facilmente personalizzabile nei gusti , proponi una variante .
       - Opzione B (Ricetta Simile): Se il piatto è una preparazione tradizionale strutturata (es. Arancini, Carbonara, Lasagne), usa gli ingredienti per trovare un piatto simile ma concettualmente diverso (es. "Supplì", "Gricia", "Cannelloni").
       
       - AZIONE 4: Dopo aver ragionato ed elaborato la nuova idea, verifica la TUA NUOVA proposta chiamando di nuovo `controlla_storico_post`.
       - Ripeti il ciclo se la nuova idea è ancora bloccata.

    3. CONCLUSIONE:
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
            ATTENZIONE: La tua idea precedente è stata BOCCIATA. 
            
            REGOLA ASSOLUTA: NON PROPORRE NESSUNO DI QUESTI TOPIC SCARTATI NELLA TUA BLACKLIST:
            [{blacklist}]
            
            Devi elaborare un'idea COMPLETAMENTE NUOVA e DIVERSA evitando la blacklist.
            
            --- REGOLE DI WORKFLOW PER LA RIGENERAZIONE ---
            1. Nel tuo PRIMO utilizzo del `think_tool`, dichiara di aver ricevuto il feedback negativo e menziona la blacklist che eviterai a tutti i costi.
            2. Usa `controlla_storico_post` per verificare la tua NUOVA idea.
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
            3. Se il nuovo topic risulta "BLOCCATO", applica la normale procedura di risoluzione conflitti (usa `ottieni_ingredienti_ricetta` e poi ragiona su Variante o Ricetta Simile).
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
        # 4. COMPILAZIONE DEL PROMPT FINALE
        prompt = SystemMessage(content=testo_prompt)
    # ==========================================================
    # 5. GESTIONE DELL'ESECUZIONE E ROUTING
    # ==========================================================

    # CASO A: Ritorno da un Tool (il ragionamento è in corso)
    if messaggi and isinstance(messaggi[-1], ToolMessage):
        risposta_llm = llm_con_tools.invoke([prompt] + messaggi)
        return {"messages": [risposta_llm], "nodo_chiamante": "planner"}

    # CASO B: Ritorno da Feedback Umano (I messaggi sono vuoti)
    elif feedback and not messaggi:
        print("   [Planner] Riavvio agente con innesco dinamico...")

        # Testo di innesco dinamico potenziato
        if "rigenera" in feedback:
            innesco = (
                "L'utente ha rifiutato la tua ultima proposta. \n"
                "⚠️ LA TUA PRIMA AZIONE IN ASSOLUTO DEVE ESSERE CHIAMARE IL `think_tool`. \n"
                "Nel `think_tool` DEVI scrivere testualmente: 'Non proporrò i seguenti piatti perché sono nella blacklist: "
                f"[{blacklist}]'. Solo dopo aver scritto questo, puoi iniziare a elaborare 3 idee COMPLETAMENTE DIVERSE."
            )
        else:
            innesco = (
                "L'utente ha richiesto una modifica specifica. \n"
                "⚠️ LA TUA PRIMA AZIONE IN ASSOLUTO DEVE ESSERE CHIAMARE IL `think_tool`. \n"
                "Nel `think_tool` DEVI confermare di aver capito la modifica e ribadire che ignorerai questi piatti: "
                f"[{blacklist}]. Non usare altri tool prima di aver fatto questo."
            )

        messaggi_da_inviare = [
            prompt,
            HumanMessage(content=innesco),
        ]
        risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)

        return {"messages": [risposta_llm], "nodo_chiamante": "planner"}

    # CASO C: Primissimo avvio - Pianificazione Automatica
    elif input == "PIANIFICAZIONE_AUTOMATICA":
        messaggi_da_inviare = [
            prompt,
            HumanMessage(content="Inizia la ricerca..."),
        ]
        risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)
        return {"messages": [risposta_llm], "nodo_chiamante": "planner"}

    # CASO D: Primissimo avvio - Post Singolo
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
    print(f"{topic}")
    reasoning_trace = state.get("reasoning_trace", [])
    messaggi = state.get("messages", [])

    testo = f"""
    Sei un Agente Investigatore esperto, specializzato in recupero dati culinari.
    Il tuo obiettivo è raccogliere la ricetta completa per il topic: '{topic}' e TUTTE le sue eventuali sottoricette.

    ### L'ALGORITMO DI RICERCA (DA SEGUIRE IN LOOP)
    Per OGNI elemento che devi cercare (partendo dal topic principale '{topic}', e applicando poi la stessa identica logica a ogni singola sottoricetta che trovi), esegui questa esatta sequenza:

    ▶ PASSO 1: RICERCA LOCALE (DB FIRST)
    - Usa `get_ingredienti_per_variante` (per interrogare il Knowledge Graph) e poi `cerca_ricetta_nel_db` cercando il VERO NOME dell'elemento corrente.
    - VALUTAZIONE: I risultati del DB contengono la ricetta ESATTA e COMPLETA (ingredienti e procedimento)?
        - SE SÌ (Trovata in DB): Il DB è la Verità Assoluta. TI È VIETATO usare il web per questa ricetta. Salta il Passo 2 e vai direttamente al PASSO 3.
        - SE NO (Assente o Incompleta): La ricerca locale è fallita. Passa al PASSO 2.

    ▶ PASSO 2: IL FILTRO DELLA RETROSPETTIVA (CORTOCIRCUITO O WEB)
    Se il PASSO 1 ha fallito, ti è SEVERAMENTE VIETATO invocare subito il tool 'esegui_ricerca_web'.
    - Rileggi e controlla attentamente il documento della Ricetta Principale che hai ricevuto precedentemente.
    - Contiene già gli ingredienti , le dosi e il procedimento per la sottoricetta corrente (es. se spiega già come fare la glassa o la pasta biscotto)?
        - SE SÌ (Sottoricetta già inclusa): Il Cortocircuito è ATTIVATO. TI È TASSATIVAMENTE VIETATO chiamare `esegui_ricerca_web`. Utilizza il 'think_tool' e dichiara: "CORTOCIRCUITO: La sottoricetta di [Nome] è già interamente presente e descritta nel testo principale in memoria." e PASSA DIRETTAMENTE al PASSO 3.
        - SE NO (Mancante o solo citata): Solo adesso SEI AUTORIZZATO ad invocare `esegui_ricerca_web` usando come query il nome specifico della sottoricetta corrente. Fatto ciò, PASSA al PASSO 3.

    ▶ PASSO 3: GESTIONE SOTTORICETTE E ASTRAZIONE (RICORSIONE)
    - Leggi attentamente gli ingredienti e il procedimento della ricetta appena acquisita (dal DB, dal Web o tramite cortocircuito).
    - Cerca eventuali SOTTORICETTE nascoste o esplicite applicando questi due criteri:
        1. CRITERIO ESPLICITO: Nei documenti del DB, se trovi la dicitura esatta "(vedi preparazione base)", sei sicuro che quella è una sottoricetta.
        2. CRITERIO DEDUTTIVO (ASTRAZIONE): Nei documenti del WEB e in alcuni casi anche nel DB, identifica la presenza di una sottoricetta deducendola dal procedimento (es. preparazioni complesse come besciamella,pastella, maionese, ragù, crema, glassa a specchio,glassa, ganache, pasta biscotto ecc..).
    - SE NE TROVI: 
        - Astrai il VERO NOME della preparazione (es. se leggi "preparare il ripieno di carne", il vero nome è "Ragù").
        - Considera questo VERO NOME come un nuovo topic pendente e RIPARTI IMMEDIATAMENTE DAL PASSO 1 per cercarlo (iniziando dal DB locale).
    - SE NON NE TROVI: La ricerca per questo specifico ramo è conclusa.

    ### REGOLE DI FERRO E DIVIETI (PENA IL FALLIMENTO)
    1. PENSIERO OBBLIGATORIO: Prima di chiamare qualsiasi tool, chiama il 'think_tool'. Spiega a che punto sei dell'algoritmo. Concludi sempre con "STATO: CONTINUO". Usa "STATO: FINITO" SOLO QUANDO hai risolto l'intera struttura (Ricetta principale + Tutte le sottoricette pendenti).
    2. DIVIETO DI COMPROMESSO: Non accettare mai risultati parziali. Se cerchi un sugo o una crema e il DB ti dà solo un "soffritto" o un "brodo", la ricerca locale è FALLITA. Devi passare al PASSO 2.
    3. DIVIETO DI FUSIONE: Se il web restituisce più ricette, NON UNIRLE MAI.
    4. DIVIETO DI SCRITTURA: Non elencare mai gli ingredienti o i procedimenti nel tuo ragionamento.
    5. REGOLA ANTI-LOOP: Se una ricerca per una sottoricetta fallisce sia nel DB che sul Web (e non è presente nel monolite), dichiaralo nel think_tool e abbandona quel ramo senza riprovare all'infinito.
    6. ASSOLUTO DIVIETO DI AMNESIA (CRITICO): Quando ti sposti su una sottoricetta, non ignorare i messaggi passati della chat. Il testo della ricetta principale è ancora lì. Rileggilo sempre al PASSO 2 per vedere se contiene già la soluzione, evitando di fare ricerche web ridondanti.
    ### STATO ATTUALE
    Sei pronto ad agire. Inizia invocando il 'think_tool' per pianificare la prima mossa sul topic principale: '{topic}'.
    """

    prompt = SystemMessage(content=testo)
    if not messaggi:
        messaggi_da_inviare = [
            prompt,
            HumanMessage(content=f"Inizia la ricerca per il topic: {topic}"),
        ]

    else:

        if reasoning_trace:

            ultime_riflessioni = [
                r for r in reasoning_trace if r.startswith("[RESEARCH]")
            ]

            riepilogo = "\n".join(
                f"- {r.replace('[RESEARCH] ', '')}" for r in ultime_riflessioni
            )

            contesto_trace = f"""
            ### RIEPILOGO DEL TUO RAGIONAMENTO FINORA
            Queste sono le tue ultime riflessioni su questo topic.
            Usale per non ripetere azioni già fatte e per decidere il passo successivo:
            {riepilogo}
            """
        else:
            contesto_trace = ""

        prompt_aggiornato = SystemMessage(content=testo + "\n" + contesto_trace)
        messaggi_da_inviare = [prompt_aggiornato] + messaggi

    risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)

    return {"messages": [risposta_llm], "nodo_chiamante": "research"}


def validator_node(state: Blog_Cucina):
    print("\n--- [NODO 3: VALIDATORE (Fact-Checking Incrociato)] ---")
    topic = state["topic_corrente"]
    tracce_di_ragionamento = state.get("reasoning_trace", [])
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
        Sei un validatore esperto di ricettari e un risolutore di dipendenze. 
        Ti sono stati forniti ESATTAMENTE {totale_doc} documenti in totale (tra DB locale e WEB).

        === L'ALGORITMO DI VALIDAZIONE (DA SEGUIRE PASSO PASSO) ===
        Nel tuo ragionamento ('think_tool'), devi eseguire RIGOROSAMENTE queste 3 fasi in ordine cronologico:

        ▶ FASE 1: ELEZIONE DELLA RICETTA PRINCIPALE
        - Cerca tra i documenti tutti quelli che descrivono il topic principale: '{topic}'.
        - Se trovi più versioni per '{topic}', confronta i loro punteggi (score web) o l'autorevolezza.
        - Eleggi la versione MIGLIORE IN ASSOLUTO. Questa diventa la "Ricetta Madre".
        - Assegna SCORE 1 alla Ricetta Madre.
        - Assegna SCORE 0 a tutte le altre versioni scartate (duplicati inferiori).

        ▶ FASE 2: ANALISI DELLE DIPENDENZE
        - Leggi attentamente gli ingredienti e il procedimento ESCLUSIVAMENTE della "Ricetta Madre" 
            appena eletta e i tuoi vecchi ragionamenti '{tracce_di_ragionamento}' per TROVARE ed ESTRARRE le sue eventuali SOTTORICETTE.
        - Dichiara chiaramente quali sono le sottoricette richieste per questa specifica preparazione.

        ▶ FASE 3: VALIDAZIONE A CASCATA (FILTRO SOTTORICETTE)
        - Ora valuta tutti i restanti documenti.
        - SE un documento descrive una Sottoricetta Necessaria (individuata nella Fase 2):
            - Se è unica, assegnale SCORE 1.
            - Se ci sono più versioni della stessa sottoricetta, eleggi la migliore (assegnando SCORE 1 alla vincitrice e SCORE 0 ai duplicati).
        - SE un documento descrive un piatto irrilevante OPPURE una sottoricetta NON richiesta dalla Ricetta Madre (es. il documento descrive un "Brodo", ma la Ricetta Madre eletta non usa brodo): assegna TASSATIVAMENTE SCORE 0.

        === FORMATO OUTPUT OBBLIGATORIO NEL RAGIONAMENTO ===
        Il tuo ragionamento finale DEVE contenere ESATTAMENTE {totale_doc} righe di valutazione. Non saltare o accorpare nessun documento.
        Per ogni documento scrivi esattamente così:
        - ID_DOC [TITOLO ESTRATTO]: Score X - Motivo: ... (es. "Eletta come Ricetta Madre", "Sottoricetta necessaria per la Ricetta Madre", "Scartata perché duplicato inferiore", "Scartata perché non richiesta dalla Ricetta Madre").

        === DOCUMENTI DA VALUTARE ===
        [DB LOCALE]
        {testo_db}

        [WEB]
        {testo_web}

        Concludi con:
        STATO: FINITO
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
    Sulla base di questa tua analisi logica:
    {riflessione_testo}

    Produci il verdetto strutturato per '{topic}'.
    ATTENZIONE (CRITICO): Devi estrarre e mappare TUTTI i documenti a cui hai assegnato SCORE 1 nella tua analisi.
    Puoi (e DEVI) approvare UNO O PIÙ  documenti (sia DB che Web) se uno rappresenta la Ricetta Madre e gli altri sono le Sottoricette necessarie.
    Nel caso in cui nessun documento riceva score 1 , i dati sono tutti inconsistenti per scrivere il post.
    - is_valid: True se i dati totali approvati sono sufficienti per scrivere il post.
    - usa_db_locale: True se almeno un documento del DB è stato approvato (Score 1).
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

    # Sfruttiamo il tuo reducer (operator.add) che ha già accumulato i pensieri
    lista_tracce = state.get("reasoning_trace", [])

    if lista_tracce:
        traces_formattate = "\n".join([f"- {t}" for t in lista_tracce])
    else:
        traces_formattate = "Nessuna traccia di ragionamento precedente registrata."

    # Consolidiamo i testi delle fonti approvate dal Validatore
    testi_approvati = "\n\n".join(dati_db_locale + dati_web)
    if not testi_approvati.strip():
        testi_approvati = "ERRORE: Nessun documento approvato dal Validatore."

    istruzione_correzione = (
        f"\n MODIFICHE PRIORITARIE RICHIESTE DAL REDATTORE:\n{feedback}\n"
        if feedback
        else ""
    )

    prompt = f"""
    Sei un Food Blogger professionista e un Redattore Editoriale senior.
    Il tuo compito è scrivere la bozza finale strutturata per l'articolo: '{topic}' (definita come RICETTA MADRE).

    Per farlo, ti vengono forniti due asset fondamentali che costituiscono il tuo unico perimetro di verità:
    1. I TESTI APPROVATI: I documenti testuali contenenti la ricetta madre e le sottoricette.
    2. LE TRACCE DI RAGIONAMENTO: L'analisi logica che mostra come il sistema ha scomposto il piatto e risolto le dipendenze.

    ========================================================================
    CRITERI DI COMPILAZIONE TASSATIVI (ZERO ALLUCINAZIONI)
    ========================================================================
    1. RIGORE STRUTTURALE (COESIONE RICETTA -> SOTTORICETTE):
       - Il tuo output deve rispecchiare fedelmente l'albero delle dipendenze stabilito nelle tracce di ragionamento.
       - Se il ragionamento precedente indica che un ingrediente (es. Besciamella, Ragù) è una SOTTORICETTA autonoma da sviluppare, DEVI mapparla interamente nella lista delle 'sotto_ricette', estraendo i suoi ingredienti specifici dal testo della fonte.
       - È SEVERAMENTE VIETATO lasciare una sottoricetta complessa descritta come semplice stringa piatta negli ingredienti diretti della ricetta madre.

    2. VINCOLO DI ANCHORING METRICOTESTUALE:
       - Estrai dosi, pesi, ingredienti e passaggi ESCLUSIVAMENTE dai testi approvati forniti. 
       - Non inventare ingredienti accessori, non arrotondare le dosi e non inserire passaggi non scritti nei documenti.
       - Se un ingrediente manca di quantità nel testo, scrivi "q.b." o "quantità non specificata" come da fonte.

    3. REGOLA ANTI-SOPRASTRUTTURE PER CONDIMENTI:
       - Non creare oggetti SottoRicetta per ingredienti pronti o topping che non subiscono una trasformazione termica o meccanica congiunta (es. parmigiano grattugiato da spolverare, prosciutto pronto). Questi vanno inseriti negli ingredienti diretti.

    4. DETTAGLIO DELLA PREPARAZIONE:
       - Genera l'elenco dei passaggi ('preparazione') in modo esteso, analitico e sequenziale. Non riassumere i procedimenti complessi in un solo paragrafo. Ogni passaggio deve essere una stringa chiara e indipendente nella lista.
    {istruzione_correzione}
    
    ========================================================================
    CONTESTO E FONTI PER LA COMPILAZIONE
    ========================================================================
    
    [TRACCE DI RAGIONAMENTO PRECEDENTE (LOG DEL GRAPH STATE)]
    {traces_formattate}

    [TESTI DELLE FONTI APPROVATE (RICETTA MADRE + SOTTORICETTE)]
    {testi_approvati}
    """

    # 3. Chiamata LLM Strutturata (Pydantic)
    llm_writer = llm.with_structured_output(RecipeDraft)
    draft = llm_writer.invoke([HumanMessage(content=prompt)])

    # 3. MIDDLEWARE DI PULIZIA PYTHON (DEDUPLICAZIONE LOGICA BINDING)
    nomi_sottoricette = set()
    for sub in draft.sotto_ricette:
        nomi_sottoricette.add(sub.classe_astratta.lower())
        nomi_sottoricette.add(sub.nome_specifico.lower())

    ingredienti_diretti_puliti = []
    for ing in draft.ingredienti_diretti:
        ing_nome_basso = ing.nome.lower()
        is_subrecipe = any(nome_sub in ing_nome_basso for nome_sub in nomi_sottoricette)
        if not is_subrecipe:
            ingredienti_diretti_puliti.append(ing)

    draft.ingredienti_diretti = ingredienti_diretti_puliti

    print("\n===== DEBUG WRITER =====")
    print(draft.model_dump())
    print("========================\n")

    # 4. Generazione Markdown deterministica tramite la funzione esterna
    markdown_finale = genera_markdown_bozza(draft)

    return {
        "recipe_draft": draft,
        "post_draft": markdown_finale,
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
