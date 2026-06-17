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
    messaggi = state.get("messages", [])

    prompt = f"""
    Sei un Agente Investigatore esperto, specializzato in recupero dati culinari.
    Il tuo obiettivo corrente è raccogliere dati completi per il topic: '{topic}'.

    ### PROTOCOLLO OPERATIVO (OBBLIGATORIO)
    Il tuo flusso di lavoro deve essere ciclico e atomico. PER OGNI SINGOLA AZIONE che intraprendi, DEVI seguire questo schema rigoroso:
    
    1. **PENSIERO (THINK):** Prima di intraprendere qualisiasi azione e chiamare qualsiasi tool, devi invocare il 'think_tool' e devi:.
       - Spiegare quale azione stai per intraprendere.
       - Spiegare PERCHÉ questa azione è necessaria (es. "Il DB non ha dati, passo al web" o "Ho trovato una sottoricetta, vado a controllare se è presente nel DB").
       - Se stai eseguendo una combinazione tra una ricetta trova online e una ricetta nel db motiva il perche della sottoricetta.
       - Concludi SEMPRE la tua riflessione con "STATO: CONTINUO" (se non hai ancora finito) o "STATO: FINITO" (se hai raccolto tutto).
    
    2. **FASI DI LAVORO(ACT):** 
       - Per completare il tuo task, segui RIGOROSAMENTE questo flusso logico:

    FASE A: RICERCA PRINCIPALE (Local-First)
    1. La tua PRIMA AZIONE in assoluto deve essere interrogare il DB locale usando il tool `cerca_ricetta_nel_db` per il topic attuale: '{topic}'.
    2. Valuta i dati ottenuti. I dati si considerano SUFFICIENTI solo se possiedi:
       - Una lista completa di ingredienti con le relative dosi.
       - Un procedimento chiaro e strutturato.
    3. SE i dati del DB locale sono SUFFICIENTI: Il DB rappresenta la tua Fonte di Verità Assoluta. TI È VIETATO usare la ricerca Web. Passa direttamente alla Fase C.
    4. SE i dati del DB locale sono ASSENTI o INSUFFICIENTI: Sei autorizzato a invocare il tool `esegui_ricerca_web` per ottenere la ricetta da internet.

    FASE B: GESTIONE SOTTORICETTE (Ricette Complesse)
    Se la ricetta che hai appena trovato (sia essa dal DB o dal Web) richiede una preparazione base aggiuntiva o una sottoricetta (es. Besciamella per le lasagne, Pasta frolla per una crostata, Maionese per un panino):
    1. DEVI obbligatoriamente fare una nuova chiamata a `cerca_ricetta_nel_db` per cercare quella specifica sottoricetta.
    2. SE la sottoricetta è presente nel DB: Usa rigorosamente gli ingredienti e il procedimento della sottoricetta locale e combinali con la ricetta principale.
    3. SE la sottoricetta NON è presente nel DB: Sei autorizzato a usare `esegui_ricerca_web` per cercare anche la sottoricetta online.

    FASE C: CONCLUSIONE
    Quando hai raccolto tutti i dati necessari (ingredienti completi e procedimento di ricette ed eventuali sottoricette), dichiara lo STATO: FINITO.
              
    ### REGOLE DI FERRO
    - NON saltare mai il 'think_tool'. L'assenza di riflessione prima di un'azione è considerata un errore grave.
    - Tu sei l'Agente ricercatore dei dati. Il tuo unico scopo è trovare le ricette. NON DEVI in nessun caso cercare di scoprire se il post è già stato pubblicato sul blog o controllare i duplicati. Questo lavoro è già stato fatto. Concentrati solo sui dati culinari.
    - Se trovi una ricetta web che cita preparazioni base (Ragù, Besciamella, ecc.), NON procedere oltre finché non avrai cercato la versione ufficiale nel DB locale.
    - Non inventare dosi o procedimenti: se la fonte non è chiara, usa 'think_tool' per dichiarare l'insufficienza dei dati.
    - La tua missione termina SOLO quando hai il set completo (Ricetta Principale + eventuali Basi Ufficiali) e hai dichiarato "STATO: FINITO".
    
    ### STATO ATTUALE
    Sei pronto ad agire. Inizia invocando il 'think_tool' per pianificare la prima mossa su '{topic}'.
    """

    if not messaggi:
        messaggi_da_inviare = [
            prompt,
            HumanMessage(content=f"Inizia la ricerca per il topic: {topic}"),
        ]
    else:
        messaggi_da_inviare = [prompt] + messaggi

    risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)

    return {"messages": [risposta_llm], "nodo_chiamante": "research"}


def validator_node(state: Blog_Cucina):
    print("\n--- [NODO 3: VALIDATORE (Fact-Checking Incrociato)] ---")

    topic = state["topic_corrente"]
    dati_db_locale = state.get("rag_documents", [])
    dati_web_grezzi = state.get("web_documents", [])
    print(f"{dati_db_locale}")
    print(f"{dati_web_grezzi}")

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
