from datetime import datetime, timedelta
from config import GIORNI_TRA_POST
from langgraph.graph import END
from langgraph.types import Command
from langchain_core.messages import (
    RemoveMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
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
    topic_originale = state.get("topic_originale", "")
    topic = state.get("topic_corrente", "")
    messaggi = state["messages"]
    feedback = state.get("human_feedback")

    lista = state.get("blacklist_topics") or []
    blacklist = ", ".join(lista) if lista else "Nessun topic scartato finora"

    if input == "PIANIFICAZIONE_AUTOMATICA":

        testo_prompt = f"""
        # Ruolo
    Sei un planner editoriale per un blog di cucina dove vengono pubblicati dei post sulla preparazione di piatti. 
    [VINCOLO TASSATIVO]: È VIETATO usare `esegui_ricerca_web` o tool RAG. Puoi usare SOLO `controlla_storico_post`, `get_ultimi_post` e `get_ingredienti` (per il Knowledge Graph).
    
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
    Usa i claim restituiti associati per:
    - Identificare lacune tematiche e vuoti all'interno del blog 
       - es. se ci sono post e claim su primi piatti, proponi un antipasto, un secondo, un contorno o un dolce cosi diversifichiamo la tipologia di ricette trattate.
       - es. se ci sono post e claim su piatti a base di carne, proponi un piatto a base di pesce o vegano cosi diversifichiamo la tipologia di proteine trattate.
       - es. se ci sono post e claim su piatti fritti, proponi un piatto al forno o al vapore cosi diversifichiamo la tipologia di cottura trattate.
       - es. se ci sono post e claim su piatti caldi, proponi un piatto freddo o a temperatura ambiente cosi diversifichiamo la tipologia di temperatura trattate.
    - diversificare le tecniche di cottura, gli ingredienti principali e le tipologie di piatti per evitare ripetizioni.
    - Giustifica anche l'ordine in cui generi i topic, se il primo è un antipasto, il secondo un primo e il terzo un dolce, spiega perché hai scelto quest'ordine.
     Usa il `think_tool` per riflettere sui risultati. Nel `think_tool`, commenta esplicitamente come i claim hanno influenzato la scelta dei topic.
    
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
            Il topic DEVE essere il nome ESPLETO di una ricetta reale e riconoscibile.
            - È ASSOLUTAMENTE VIETATO ❌ combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
            - È VIETATO USARE nomi generici come "dolce", "antipasto", "primo", "contorno", ecc.
            **Devi usare un nome specifico che identifichi un piatto preciso, ad esempio:**
            - ❌ SBAGLIATO (VIETATO): "sgombro al forno con patate" → usa "Filetto di sgombro al forno" 
            - ❌ SBAGLIATO (VIETATO): "dolce al caffè" → usa "Tiramisù" o "Panna cotta al caffè"
            - ❌ SBAGLIATO (VIETATO): "antipasto" → usa "Bruschette al pomodoro" o "Caprese"
            - ❌ SBAGLIATO (VIETATO): "primo a base di pesce" → usa "Spaghetti alle vongole"
            - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate" usa "pollo al limone"
            - ❌ SBAGLIATO (VIETATO): "Filetto di manzo alla griglia con patate al forno" → usa "Filetto di manzo alla griglia" 
            
            - ✅ CORRETTO: "Pollo al limone"
            - ✅ CORRETTO: "Verdure grigliate"
            - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            - ✅ CORRETTO: "patate al forno"
            
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
            REGOLA ANTI-STALLO (attiva solo se la blacklist contiene già 1 o più piatti):
                Quando la blacklist non è vuota, significa che le tue proposte precedenti sono state rifiutate perché troppo simili al piatto originale. In questo caso:
                - NON proporre MAI un piatto che contenga l'ingrediente principale del piatto bloccato nella stessa forma (es. se il piatto bloccato è "Pollo al limone", evita QUALSIASI piatto a base di pollo intero, a fette o a cubetti).
                - Spostati su una proteina DIVERSA (es. vitello, maiale, tacchino, pesce) oppure su un piatto completamente diverso che condivida solo la tecnica o il sapore (es. un risotto al limone, una pasta al limone, un dessert al limone).
                - L'obiettivo è sorprendere il lettore con qualcosa di affine ma inaspettato, non una ripetizione mascherata.
            REGOLA ASSOLUTA: NON PROPORRE NESSUNO DI QUESTI TOPIC SCARTATI. 
                ⛔Devi generare 3 idee COMPLETAMENTE DIVERSE DA :{blacklist}.
            
            --- REGOLE DI WORKFLOW PER LA RIGENERAZIONE ---
            1. Nel tuo PRIMO utilizzo del `think_tool` in questo nuovo ciclo, DEVI dichiarare esplicitamente di aver ricevuto il feedback negativo e menzionare i topic scartati ({blacklist}) che eviterai.
            2. Chiama il tool `get_ultimi_post` per capire cosa è stato pubblicato di recente. (NON RICHIAMARLO PIù DI UNA VOLTA).
            3. Continua a usare il `think_tool` dopo ogni controllo con `controlla_storico_post`. 
            4. Usa la dicitura "STATO: CONTINUO" finché non trovi 3 nuove idee approvate dal tool.
            5. Usa la dicitura "STATO: FINITO" solo alla fine, quando hai i 3 topic definitivi.
            
             ## Formato dei topic (REGOLA FERREA)
            **Il topic DEVE essere il nome ESPLETO di una ricetta reale e riconoscibile.** 
            - È ASSOLUTAMENTE VIETATO❌combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
            - **È VIETATO USARE nomi generici come "dolce", "antipasto", "primo", "contorno", ecc.**
            **Devi usare un nome specifico che identifichi un piatto preciso, ad esempio:**
            - ❌ SBAGLIATO (VIETATO): "dolce al caffè" → usa "Tiramisù" o "Panna cotta al caffè"
            - ❌ SBAGLIATO (VIETATO): "antipasto" → usa "Bruschette al pomodoro" o "Caprese"
            - ❌ SBAGLIATO (VIETATO): "primo a base di pesce" → usa "Spaghetti alle vongole"
            - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
            - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
            - ❌ SBAGLIATO (VIETATO): "primo a base di pesce"
            - ✅ CORRETTO: "Pollo al limone"
            - ✅ CORRETTO: "Verdure grigliate"
            - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            - NON cercare dati online o in locale. Il tuo compito è solo pianificare.
            
           [VINCOLO TASSATIVO]: È VIETATO usare `esegui_ricerca_web` o tool RAG. Puoi usare SOLO `controlla_storico_post` e `get_ingredienti` (per il Knowledge Graph).
        
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
            **Il topic DEVE essere il nome ESPLETO di una ricetta reale e riconoscibile.** 
            - È ASSOLUTAMENTE VIETATO ❌ combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
            - **È VIETATO USARE nomi generici come "dolce", "antipasto", "primo", "contorno", ecc.**
            **Devi usare un nome specifico che identifichi un piatto preciso, ad esempio:**
            - ❌ SBAGLIATO (VIETATO): "dolce al caffè" → usa "Tiramisù" o "Panna cotta al caffè"
            - ❌ SBAGLIATO (VIETATO): "antipasto" → usa "Bruschette al pomodoro" o "Caprese"
            - ❌ SBAGLIATO (VIETATO): "primo a base di pesce" → usa "Spaghetti alle vongole"
            - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
            - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
            - ❌ SBAGLIATO (VIETATO): "primo a base di pesce"
            - ✅ CORRETTO: "Pollo al limone"
            - ✅ CORRETTO: "Verdure grigliate"
            - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            - NON cercare dati online o in locale. Il tuo compito è solo pianificare.
            
            [VINCOLO TASSATIVO]: È VIETATO usare `esegui_ricerca_web` o tool RAG. Puoi usare SOLO `controlla_storico_post` e `get_ingredienti` (per il Knowledge Graph).
            """

        prompt = SystemMessage(content=testo_prompt)

    else:
        testo_prompt = """
    Sei il Direttore Editoriale di un blog di cucina. Analizza la direttiva dell'utente, assicurandoti che non ci siano duplicati.

    REGOLE DEL FLUSSO AUTONOMO E RISOLUZIONE CONFLITTI (ReAct):
     FASE 1. VERIFICA INIZIALE: Usa SEMPRE `controlla_storico_post` per verificare se la direttiva dell'utente è già stata pubblicata.
             APPENA RICEVI LA RISPOSTA DEL TOOL, DEVI chiamare il `think_tool` e dichiarare il risultato (OK o BLOCCATO). Solo dopo puoi procedere.
    
     FASE 2. PROCEDURA Di RAGIONAMENTO  (SEGUI ALLA LETTERA):
       ⚠️ REGOLA FERREA: 
            -In caso di OK, NON chiamare MAI `get_ingredienti` né `controlla_storico_post` per varianti.
            Devi IMMEDIATAMENTE andare alla FASE 3 e concludere con "STATO: FINITO" senza ulteriori azioni.
            -Se il database risponde "BLOCCATO",  direttiva dell'utente è già stata pubblicata. NON generare idee a caso. Esegui questa sequenza esatta SOLO SE il database risponde "BLOCCATO":
       - AZIONE 1: Chiama immediatamente il tool `get_ingredienti` passando il nome del piatto bloccato.
       - AZIONE 2: chiama il tool `get_ultimi_post` per recuperare gli ultimi post publicati 
       - AZIONE 3: Attendi i risultati da Neo4j (ingredienti e ultimi post).
       - AZIONE 4: Chiama il tuo `think_tool` per ragionare sugli ingredienti estratti e scegli UNA proposta tra le due opzioni seguenti.
                     Il tuo obiettivo sarà proporre una nuova ricetta che vada a coprire eventiali gap presenti nel tuo dominio. Analizza gli ultimi post pubblicati in modo da diversificare il blog non proponendo sempre la stessa cosa(categoria della ricetta, metodi di cottura). Scegli l'opzione in base alla NATURA del piatto bloccato:
                    ▶ OPZIONE A – VARIANTE GUSTO / COTTURA (scegli questa se il piatto bloccato è una base neutra facilmente declinabile, es. pizza, risotto, pasta al sugo, torta):
                        - Cambia il GUSTO o il CONDIMENTO principale, NON la proteina base.
                        - Esempio: se il piatto bloccato è “Pizza margherita”, proponi “Pizza capricciosa” o “Pizza ai quattro formaggi”.
                        - Esempio: se è “Risotto ai funghi”, proponi “Risotto allo zafferano” o “Risotto al radicchio”.
                        - VIETATO: cambiare solo il tipo di carne/pesce mantenendo identica la preparazione (es. “Pollo al limone” → “Vitello al limone” è PIGRIZIA, NON farlo).
                    ▶ OPZIONE B – RICETTA SIMILE (scegli questa se il piatto bloccato è una preparazione strutturata e completa, es. Arancini, Carbonara, Lasagne, Pollo al limone, Spezzatino):
                        - Individua un piatto DIVERSO che condivida 1-2 ingredienti chiave ma abbia un NOME, una TECNICA e un'IDENTITÀ CULINARIA PROPRIA.
                        - Esempi REALI:
                            - “Pollo al limone” → “Scaloppine al limone” o “Piccata di vitello al limone” (cambia taglio, tecnica, nome).
                            - “Carbonara” → “Gricia” o “Cacio e pepe” (stessa famiglia, identità distinte).
                            - “Lasagne al ragù” → “Cannelloni ricotta e spinaci” o “Pasticcio di maccheroni”.
                        - VIETATO: riproporre lo stesso piatto con un ingrediente cambiato (es. “Pollo al limone” → “Tacchino al limone” è bandito).
                    In entrambi i casi, la proposta DEVE essere un piatto COMPLETO e RICONOSCIBILE, non una combinazione artificiale di ingredienti.
        - AZIONE 4: Dopo aver ragionato ed elaborato la nuova idea, verifica la TUA NUOVA proposta chiamando di nuovo `controlla_storico_post`.
        - Ripeti il ciclo se la nuova idea è ancora bloccata. SE una tua idea riceve "OK" vai alla FASE 3
    
     FASE 3. CONCLUSIONE:
       -Solo quando avrai ottenuto un "OK" per un topic, termina con "STATO: FINITO".
       + Nota: Il topic verrà comunque sottoposto all'approvazione dell'utente, ma il tuo compito è terminato..
       
        --- FORMATO DEI TOPIC (REGOLA FERREA) ---
            - Singola Preparazione: Il topic deve essere una ricetta reale, specifica e SINGOLA. 
            - È ASSOLUTAMENTE VIETATO combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
              - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
              - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
              - ❌ SBAGLIATO (VIETATO): "primo a base di pesce"
              - ✅ CORRETTO: "Pollo al limone"
              - ✅ CORRETTO: "Verdure grigliate"
              - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            
    [VINCOLO TASSATIVO]: È VIETATO usare `esegui_ricerca_web` o tool RAG. Puoi usare SOLO `controlla_storico_post` e `get_ingredienti` (per il Knowledge Graph).
    """

        # 2. GESTIONE DEL FEEDBACK: RIGENERA TOTALE CON BLACKLIST
        if feedback == "rigenera":
            print(f"   [Planner] Rilevato 'rigenera' singolo. Blacklist: {blacklist}")
            testo_prompt += f"""
            
            --- FEEDBACK UTENTE (PROPOSTA SINGOLA RIFIUTATA) ---
            ATTENZIONE: La tua idea precedente è stata BOCCIATA e si trova dentro la  {blacklist} . 
            
            REGOLA ASSOLUTA: NON PROPORRE NESSUNO DI QUESTI TOPIC SCARTATI NELLA TUA BLACKLIST perche l'utente li ha bocciati:
            [{blacklist}]
            
            Devi elaborare un'idea COMPLETAMENTE NUOVA e DIVERSA evitando i dati contenunti in:  {blacklist}.
            
            --- REGOLE DI WORKFLOW PER LA RIGENERAZIONE ---
            1. Nel tuo PRIMO utilizzo del `think_tool`, dichiara di aver ricevuto il feedback negativo e menziona la blacklist che eviterai a tutti i costi.
            2. Considera varianti o nuove ricette relative al: "{topic_originale}" che era il topic iniziale proposto dall'utente ma che è risultata già pubblicato.
            3. Usa `controlla_storico_post` per verificare ESCLUSIVAMENTE la tua NUOVA idea.
            4. Se risulta "BLOCCATO", applica la normale procedura di risoluzione conflitti.
            5. 🚨 REGOLA DI STOP: Appena ricevi un "OK" definitivo, DEVI FERMARTI IMMEDIATAMENTE e chiudere il messaggio con "STATO: FINITO".
            

            
            ## Formato dei topic (REGOLA FERREA)
            - **Singola Preparazione:** Il topic deve essere una ricetta reale, specifica e SINGOLA. 
            - È ASSOLUTAMENTE VIETATO combinare una pietanza principale con un contorno o creare piatti composti. Devi indicare solo l'elemento principale.
            - ❌ SBAGLIATO (VIETATO): "Pollo al limone con contorno di verdure grigliate"
            - ❌ SBAGLIATO (VIETATO): "Filetto di manzo con patate al forno"
            - ❌ SBAGLIATO (VIETATO): "primo a base di pesce"
            - ❌ SBAGLIATO (VIETATO): provare a riproporre "{topic_originale}"essendo un duplicato.
            - ✅ CORRETTO: "Pollo al limone"
            - ✅ CORRETTO: "Verdure grigliate"
            - ✅ CORRETTO: "Filetto di manzo al pepe verde"
            - NON cercare dati online o in locale. Il tuo compito è solo pianificare.
            non devi mai proporre nessuno di questi topic [{blacklist}] e "{topic_originale}"!!
            [VINCOLO TASSATIVO]: È VIETATO usare `esegui_ricerca_web` o tool RAG. Puoi usare SOLO `controlla_storico_post` e `get_ingredienti` (per il Knowledge Graph).
            """

        # 3. GESTIONE DEL FEEDBACK: MODIFICA GUIDATA
        if feedback and feedback.startswith("modifica:"):

            istruzioni = feedback.replace("modifica:", "").strip()

            print(
                f"   [Planner] Rilevato 'modifica' singolo. Istruzioni: {istruzioni}. Blacklist: {blacklist}"
            )

            testo_prompt += f"""
            
            --- FEEDBACK UTENTE (RICHIESTA DI MODIFICA) ---
            L'utente stava valutando il topic: "{topic}".
            Ha richiesto questa modifica specifica: "{istruzioni}".
 
            Il tuo obiettivo è trovare UN SOLO topic che applichi la modifica richiesta
            a "{topic}". Ragiona sul topic di partenza e sull'istruzione,
            poi proponi un nuovo nome di ricetta che le incorpori entrambe.
            Esempio: topic "Insalata di carote e piselli" + istruzione "aggiungi la maionese"
            → nuovo topic "Insalata di carote, piselli e maionese".
 
            REGOLE ASSOLUTE:
            1. Il NUOVO topic DEVE applicare "{istruzioni}" a "{topic}".
            2. NON PROPORRE MAI: {blacklist} — già scartati dall'utente.
            3. Alla fine devi avere esattamente 1 topic approvato.
 
            --- WORKFLOW ---
            1. Nel PRIMO `think_tool` dichiara: topic di partenza, modifica da applicare,
               nuovo topic che hai deciso di proporre.
            2. Usa `controlla_storico_post` SOLO per il NUOVO topic.
            3. Se risulta "BLOCCATO": usa `get_ingredienti` e proponi una variante.
            4. 🚨 STOP: al primo "OK" chiudi con "STATO: FINITO".
 
            # Formato (REGOLA FERREA)
            - Topic singolo e specifico — mai combinazioni con contorno.
            - ❌ VIETATO: usare "{istruzioni}" direttamente come topic.
            - ✅ CORRETTO: ricava un nome di piatto reale dalla modifica.
        [VINCOLO TASSATIVO]: È VIETATO usare `esegui_ricerca_web` o tool RAG. Puoi usare SOLO `controlla_storico_post` e `get_ingredienti` (per il Knowledge Graph).
            """

        prompt = SystemMessage(content=testo_prompt)

    # Ritorno da un Tool (il ragionamento è in corso)
    if messaggi and isinstance(messaggi[-1], ToolMessage):
        if messaggi[-1].name == "get_claim_pertinenti":
            if "NESSUN_CLAIM" not in messaggi[-1].content:
                print(
                    "[PLANNER] Claim pertinenti ricevuti e analizzati nel prossimo think_tool."
                )
            else:
                print("[PLANNER] Nessun claim pertinente trovato.")
        risposta_llm = llm_con_tools.invoke([prompt] + messaggi)
        return {"messages": [risposta_llm], "nodo_chiamante": "planner"}

    # Ritorno da Feedback Umano (I messaggi sono vuoti)
    elif feedback and not messaggi:
        print("   [Planner] Riavvio agente con innesco dinamico...")

        # in base al feedback
        if "rigenera" in feedback:
            print(f"{topic_originale}")

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
                f"la proposta originale richiesto dall'utente era:'{topic_originale} "
                "Devi restare nell'area semantica di questo piatto: proponi varianti o ricette"
                "simili che condividono ingredienti principali o tecnica di preparazione. "
                f"NON proporre MAI il topic originale stesso ('{topic_originale}') perché è già stato pubblicato"
                f"Solo dopo aver scritto questo, puoi iniziare a elaborare {idee}.\n"
                f"🚨 REGOLA DI STOP: {stop}, FERMATI IMMEDIATAMENTE con 'STATO: FINITO'."
            )

        else:  # caso modifica

            if input == "PIANIFICAZIONE_AUTOMATICA":
                idee = "1 NUOVO topic (mantenendo gli altri 2 del piano precedente)"
            else:
                idee = (
                    f"una versione modificata di '{topic}' "
                    f"applicando questa istruzione: '{istruzioni}'"
                )

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

        """ if (
            topic_proposto
            and topic_proposto != "None"
            and topic_proposto not in blacklist_attuale
        ):
            blacklist_attuale.append(topic_proposto)
        """

        messaggi_da_cancellare = [
            RemoveMessage(id=m.id) for m in state.get("messages", [])
        ]

        return Command(
            update={
                "human_feedback": f"modifica:{istruzioni}",
                "blacklist_topics": blacklist_attuale,
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

    conteggio_web = 0
    for msg in state.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.name == "esegui_ricerca_web":
            conteggio_web += 1

    testo_base = f"""
    
    ## IL TUO RUOLO
    Sei un Agente ricercatore culinario. Il tuo compito è raccogliere i dati completi per la ricetta indicata dal topic: **'{topic}'**, e per tutte le sue eventuali sottoricette che superano il test di seguito.
    non ti interessa scegliere la ricetta migliore, ma SOLO raccogliere TUTTE le informazioni disponibili.
    
🚨**LIMITE MASSIMO RICERCHE PER LA RICETTA PRINCIPALE**: hai già effettuato {conteggio_web} chiamata a `esegui_ricerca_web` per questo topic.
            -  Puoi effettuarne la ricerca della ricetta madre al massimo UNA VOLTA. Se raggiungi il limite senza aver trovato una ricetta valida, DEVI dichiarare
            - "FALLIMENTO: Ricetta non disponibile" e concludere immediatamente con STATO: FINITO, senza cercare ulteriormente.
   
   Per ogni elemento da cercare (ricetta principale o sottoricetta) l’ordine di priorità è fisso e mai derogabile:
    
    DB LOCALE – fonte primaria.
    RICETTA MADRE GIÀ IN MEMORIA – solo se il DB fallisce, e solo per le sottoricette.
    WEB – solo se DB e Ricetta Madre non forniscono una sottoricetta completa.

    È VIETATO saltare dal punto 1 al punto 3 senza aver eseguito il punto 2.
        

    PAROLE CHIAVI PER LA RICERCA:
        Se la ricetta contiene uno di questi termini la ricetta deve soddisfare la condizione corrispondente:
        -light: cerca versioni a basso contenuto calorico della ricetta.
        -vegetariana: cerca versioni senza carne o pesce.
        -vegana: cerca versioni senza ingredienti di origine animale.
        -senza glutine: cerca versioni senza glutine.
        -senza lattosio: cerca versioni senza lattosio.
        -senza zucchero: cerca versioni senza zuccheri aggiunti.
        -senza uova: cerca versioni senza uova. 
     ---
    
    
    ## FLAG DI STATO (DEVI TENERE A MENTE)
        - `main_recipe_found = False`  (all'inizio)
        - Quando ottieni una ricetta madre completa (da DB o Web), **imposta** `main_recipe_found = True` e **non eseguire più alcuna ricerca per il topic principale**.
        - Ogni volta che stai per chiamare un qualsiasi tool per la ricetta madre, **controlla** questo flag. Se è `True`, **non chiamare nessun tool** e passa direttamente alla FASE 2.
    
     ---
     
    ## TEST DELLA SOTTORICETTA (OBBLIGATORIO)
        
        *"Se preparassi questo ingrediente DA SOLO, otterrei un prodotto con un'identità culinaria propria che esiste anche fuori da questa ricetta?"*

        ✅ **SUPERATO** – l'ingrediente subisce una TRASFORMAZIONE (cottura, emulsione, montatura) che lo rende un prodotto nuovo e riconoscibile e possiede dei suoi ingredienti.
        🚨Esempi: besciamella, ragù, maionese, crema pasticcera, pasta biscotto, brodo di carne, pesto di basilico, crema al mascarpone .

        Due criteri operativi guidano la tua analisi:
        - **CRITERIO ESPLICITO**: se il testo di un ingrediente di una ricetta del DB contiene rimandi come “(vedi preparazione base)” o similari, quello è un chiaro indicatore di una sottoricetta.
        - **CRITERIO DEDUTTIVO**: se la ricetta cita nei suoi ingredienti e nei passaggi del suo procedimento, la preparazione di un ingrediente complesso (es. besciamella, ragù, crema pasticcera, pasta biscotto, pastella,brodo di carne, maionese, ecc...). Quel ingrediente va considerato come una sottoricetta.

        ❌ **FALLITO** – l'ingrediente è già pronto o subisce solo miscelazione a crudo, senza trasformazione significativa e non possiede dei suoi ingredienti. In particolare:
            - **Condimenti e mix crudi**: marinature, panature, miscele di spezie, triti semplici, salse crude.-> NO SOTTORICETTA
            - **Prodotti finiti**: formaggi (anche vegetali), salumi/affettati (anche vegetali), yogurt, paste spalmabili (pasta di olive, burro di arachidi).-> NO SOTTORICETTA
            - **Bevande e liquidi**: caffè, tè, vino, liquori, brodo già fatto,latte, acqua aromatizzata.-> NO SOTTORICETTA
            - **Preparazioni minime**: uova sode, ammolli (uvetta, funghi secchi), guarnizioni (granella, zucchero a velo).-> NO SOTTORICETTA
            - **Bagne e sciroppi dolci**: bagna di fragole, bagna al caffè, sciroppo di zucchero, succo di frutta preparato al momento.-> NO SOTTORICETTA
            - **ingredienti semplici**: Ingredienti sia crudi ma anche che subiscono una trasformazione(cottura) ma non hanno un identita culinaria propria non possono essere considerate delle sottoricette. -> NO SOTTORICETTA
             
        esempi: melanzane fritte,funghi arrostiti, peperoni grigliati, olio, parmigiano, grnana padono, cioccolato, formaggio, pasta, riso, broccoli, verdure, manzo, tonno. -> NO SOTTORICETTA
            
     ⚠️ Se il test FALLISCE, consideralo un normale ingrediente. NON cercarlo come SOTTORICETTA.
     🚨REGOLA: è SEVERAMENTE VIETATO CONSIDERARE LA RICETTA MADRE STESSA O SUE VARIANTI COME SOTTORICETTA.
     nel tuo `think_tool` se trovi una sottoricetta DEVI scrivere: ho trovato la sottoricetta x spiegando perche la consideri una sottoricetta
      ---
    
    #LOOP DELLA SOTTORICETTA
        Per ogni singola sottoricetta individuata, DEVI seguire ESATTAMENTE questo ordine (è severamente vietato saltare i passaggi):
        
        CHIAMA IL THINK TOOL DICENDO PER CHE TOPIC STAI INIZIANDO LA RICERCA.

        1. **CHECK 1 - KNOWLEDGE GRAPH (Obbligatorio e prioritario)**:
        - Chiama il tool `get_ricetta_dal_grafo` passando il nome della sottoricetta.
        - ✅ Se TROVATA: la sottoricetta è RISOLTA. Memorizza i dati ufficiali del blog. NON cercare nel Vector DB e NON cercare sul Web. Torna alla FASE 2 (se ci sono altre sottoricette) o concludi.
        - ❌ Se NON TROVATA: vai al passaggio 2.

        2. **Preparazione Query Espansa per la Sottoricetta**:
        - Chiama `get_ingredienti` e `get_claims_per_retrieval` per la sottoricetta.
        - Unisci i risultati in una query discorsiva.

        3. **CHECK 2 - VECTOR DB LOCALE**:
        - Chiama `cerca_ricetta_nel_db` passando la query espansa appena creata.
        - ✅ Se TROVATA una ricetta completa: memorizzala come RISOLTA e torna alla FASE 2.
        - ❌ Se NON TROVATA: invoca `think_tool` per spiegare il fallimento e vai al passaggio 4.

        4. **CHECK 3 - RICERCA WEB**:
        - Chiama `esegui_ricerca_web` per la sottoricetta.
        - ✅ Se TROVATA: memorizzala e torna alla FASE 2.
        - ❌ Se NON TROVATA: dichiara "FALLIMENTO SOTTORICETTA" e procedi con le altre.
        
     🚨NON ESEGUIRE IL TEST DELLA SOTTORICETTA ALLE SINGOLE SOTTORICETTE
     
     🚨UNA VOLTA TROVATA MEMORIZZA LA SOTTORICETTA E DICHIARA COCNLUSA LA RICERCA PER QUEST'UTLIMA
            -NON RIPETERE PIU LA RICERCA PER QUESTA SOTTORICETTA PURE SE PRESENTE IN SUCCESSIVE RICETTE MADRI IN QUANTO L'HAI GIA ELABORATA
     
     ⚠️ REGOLA ANTI-RITORNO: Dopo aver completato la ricerca di una sottoricetta (in uno qualsiasi degli step), torna alla FASE 2 per verificare se ci sono altre sottoricette per la STESSA Ricetta Madre. NON passare a una nuova Ricetta Madre finché non hai completato tutte le sottoricette di quella corrente.
    ---
    
    ##🚨 REGOLE GLOBALI
    
        **PENSIERO OBBLIGATORIO**: Invoca il `think_tool` prima di chiamare qualsiasi tool di ricerca (`esegui_ricerca_web`, `cerca_ricetta_nel_db` e `get_ricetta_dal_grafo`( lui solo per le sottoricette)) spiegando:
            - In quale FASE ti trovi.
            - Cosa stai per fare e perché.
            - Termina con “STATO: CONTINUO” (o “STATO: FINITO” se hai completato tutto).

        **QUERY ESPANSA**:
            - Per ogni NUOVO elemento da cercare nel DB (ricetta madre o sottoricetta), chiama `get_ingredienti` e `get_claim_per_retrieval` UNA volta.
            - Se ritenti la ricerca dello stesso elemento dopo un fallimento, riutilizza la query già ottenuta senza richiamare `get_ingredienti` e `get_claim_per_retrieval`.
            - Usa sempre la query espansa con `cerca_ricetta_nel_db`.
            
        **RICERCHE NEL WEB**: è VIETATO effettuare ricerche nel web senza prima aver fatto le rispettive ricerche nel db
        
            
        🚨 **DOPO AVER OTTENUTO UNA RICETTA MADRE COMPLETA (DA DB O WEB)**:
            - NON chiamare `get_ingredienti` o `get_claim_per_retrieval` per la ricetta madre.
            - NON CONSIDERARLA UNA SOTTORICETTA
            - Utilizza i dati già presenti nella risposta della ricerca (ingredienti e procedimento).
            - Questi strumenti vanno usati SOLO per **nuovi** elementi (sottoricette) non ancora esaminati.
                        

    ---
    
    ## ALGORITMO DI RICERCA

    ### ▶ FASE 1 – RICERCA DELLA RICETTA PRINCIPALE (TOPIC)

        1. **Ottieni query espansa**:
        - Chiama `get_ingredienti`  e `get_claim_per_retrieval` per `{topic}`.
        - effettua la ricerca nel DB.
        2. **Cerca nel DB**(OBBLIGATORIO DA FARE):
        - Usa `cerca_ricetta_nel_db` con la query espansa.
        3. **Valuta il risultato**:
        - ✅ Se TROVATA una ricetta **completa** (ingredienti + procedimento) → memorizza e passa direttamente alla **FASE 2** (è VIETATO cercare sul Web).
        - ❌ Se NON TROVATA o incompleta → chiama `think_tool` per analizzare il fallimento, poi vai al punto 4.
        ⚠️ REGOLA DI STOP RICERCHE: Una volta che hai trovato una o più ricette madri valide (da DB o Web) per il topic '{topic}', NON cercare ulteriori versioni. Passa immediatamente alla FASE 2 senza ulteriori chiamate a `esegui_ricerca_web` o `cerca_ricetta_nel_db` per la ricetta principale.
        4. **Ricerca Web (solo in caso di fallimento DB)**:
        - Usa `esegui_ricerca_web` per `{topic}` usando la query espansa.
        - Valuta i documenti ottenuti: cerca una ricetta completa (ingredienti + procedimento).
        - ✅ Se trovi uno o piu documenti completi e coerenti per `{topic}`:
            -**Imposta `main_recipe_found = True`**
            -memorizza  TUTTI I DOCUMENTI come fonte primaria 
            -Passa IMMEDIATAMENTE alla **FASE 2 ANALISI DELLE SOTTORICETTE** senza chiamare altri tool per la ricetta madre.
        - ❌ Se nessun documento è completo, coerente per `{topic}` o la ricerca non restituisce risultati → dichiara "FALLIMENTO: Ricetta non disponibile" e termina con STATO: FINITO.
    

    ### ▶ FASE 2 – ANALISI DELLE SOTTORICETTE (ESEGUI PER OGNI RICETTA MADRE)
        🚨⚠️ NON CHIAMARE MAI NESSUN TOOL USANDO IL NOME DELLA RICETTA MADRE DURANTE LA FASE 2. In questa fase il tuo scopo è trovare le sottoricette qualora ci siano.
        🚨 è SEVERAMENTE VIETATO CONSIDERARE LA RICETTA MADRE COME UNA SOTTORICETTA
    
        Per OGNI Ricetta Madre trovata, esegui questa procedura COMPLETA prima di passare alla successiva:
       
        1. Leggi attentamente ingredienti e procedimento.
        2. Applica il TEST DELLA SOTTORICETTA a ogni elemento complesso.
        3. Esito dell’analisi:
        - **NESSUNA SOTTORICETTA**: dichiaralo esplicitamente nel think_tool: "Nessuna sottoricetta per [Nome Ricetta Madre]. Passo alla prossima Ricetta Madre o concludo se non ce ne sono altre."
        - **SOTTORICETTE TROVATE**: astrai il loro VERO NOME e, per OGNUNA di esse, esegui SUBITO la FASE 3 (ricerca). Solo dopo aver completato TUTTE le sottoricette di questa Ricetta Madre, puoi passare alla Ricetta Madre successiva.
    

    ### ▶ FASE 3 – RICERCA DI UNA SOTTORICETTA (loop per ognuna)
     APPLICA IL LOOP DELLE SOTTORICETTE ad ogni singola sottoricetta.
        
        ATTENZIONE: Esegui rigorosamente i seguenti passaggi IN ORDINE per OGNI sottoricetta trovata.

        Applica integralmente il LOOP DELLA SOTTORICETTA definito sopra
        
        ⚠️ REGOLA ANTI-RITORNO: Dopo aver completato la ricerca di una sottoricetta (in uno qualsiasi degli step), torna alla FASE 2 per verificare se ci sono altre sottoricette per la STESSA Ricetta Madre. NON passare a una nuova Ricetta Madre finché non hai completato tutte le sottoricette di quella corrente.
        
    ---

    ### ▶ TERMINAZIONE
    - Se non trovi nessuna sottoricetta.
    - Quando tutte le sottoricette di una Ricetta Madre sono risolte o abbandonate, la Ricetta Madre è completa.
    - Se ci sono più Ricette Madri (più ricette sullo stesso topic), processale una alla volta.
    - Solo quando **tutto** è stato completato, invoca `think_tool` con “STATO: FINITO”.

    ---

    ## STATO ATTUALE
    Inizia invocando `think_tool` per dichiarare l'avvio della FASE 1 per il topic: **'{topic}'**.
    """

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
    print(f" [VALIDATOR] documenti web: {testo_web}")
    print(f" [VALIDATOR] documenti db: {testo_db}")

    # ── PASSO 1: messaggi vuoti → l'agente chiama get_claim_pertinenti ──
    if not messaggi:
        print("non ci sono messaggi!")
        prompt_riflessione = f"""
        Sei un Validatore Supremo. Prima di tutto, chiama `get_claim_pertinenti` per '{topic}',
        NON CHIAMARE ALTRI TOOL PRIMA DI AVER ESEGUITO QUESTO PASSO.
        """
        messaggio = [
            SystemMessage(content=prompt_riflessione),
            HumanMessage(content=f"Recupera i claim pertinenti per '{topic}'."),
        ]
        risposta = llm_con_tools.invoke(messaggio)

        return {"messages": [risposta], "nodo_chiamante": "validator"}

        # Estrai il contenuto dei claim se disponibili
    claim_text = ""
    for msg in messaggi:
        if isinstance(msg, ToolMessage) and msg.name == "get_claim_pertinenti":
            claim_text = msg.content

        else:
            claim_text = "NESSUN CLAIM DISPONIBILE"

    for msg in reversed(messaggi):
        if isinstance(msg, ToolMessage):
            ultimo_tool = msg.name
            break

    if ultimo_tool == "get_claim_pertinenti":

        print("sono il validator inizio a ragionare")
        prompt_riflessione = f"""
                Sei un Validatore Supremo esperto di ricettari e un rigoroso risolutore di dipendenze.
                Ti sono stati forniti ESATTAMENTE {totale_doc} documenti in totale
                ({len(dati_db_locale)} dal DB locale, {len(dati_web_grezzi)} dal WEB).
        
                🚨 ALLERTA ROSSA: QUESTA È L'UNICA INTERAZIONE CHE AVRAI. NON CI SARÀ NESSUN "PROSSIMO TURNO".
                DEVI ESEGUIRE TUTTE E 3 LE FASI ORA, ALL'INTERNO DI QUESTA SINGOLA CHIAMATA AL TOOL.
                USARE "STATO: CONTINUO" È UN ERRORE FATALE CHE COMPROMETTERÀ IL SISTEMA.
        
                🚨 REGOLA DI ETICHETTATURA (CRITICA, controllata a valle da codice automatico):
                - Usa l'etichetta ESATTA con cui il documento ti è stato presentato: se proviene
                dalla sezione [WEB] è "WEB_DOC_n", se proviene dalla sezione [DB LOCALE] è
                "DB_DOC_n". NON convertire mai un'etichetta nell'altra.
                - Il DB locale contiene ESATTAMENTE {len(dati_db_locale)} documenti in questo turno.
                {"Quindi NON esiste NESSUN 'DB_DOC_x': è VIETATO scrivere qualsiasi riga con prefisso DB_DOC nell'elenco finale." if len(dati_db_locale) == 0 else ""}
                - Il WEB contiene ESATTAMENTE {len(dati_web_grezzi)} documenti in questo turno.
                {"Quindi NON esiste NESSUN 'WEB_DOC_x': è VIETATO scrivere qualsiasi riga con prefisso WEB_DOC nell'elenco finale." if len(dati_web_grezzi) == 0 else ""}
        
                Ecco i claim pertinenti che hai appena recuperato:
                {claim_text}
        
                analizza i claim e confrontali con i documenti, controlla se sono pertinenti o meno ai documenti trovati e motiva la tua riposta 
                usa il `think_tool` per esprimere la tua valutazione COMPLETA  NEL CAMPO 'analisi_contesto' .
        
                Compila i 3 campi del `think_tool` SEGUENDO TASSATIVAMENTE QUESTE ISTRUZIONI:
        
                ▶ NEL CAMPO 'analisi_contesto' (ESEGUI FASE 1 e FASE 2 INSIEME ADESSO):
        
                - Confronta TUTTI i documenti che trattano '{topic}'.
                - Eleggi il MIGLIORE come "Ricetta Madre" (assegnali SCORE 1) in base al punteggio o all'autorevolezza.
                - Assegna SCORE 0 a tutti gli altri documenti che parlano dello topic o di sue varianti '{topic}' (sono duplicati inferiori).
                - Se non trovi nessun documento pertinente per il '{topic}' dichiara fallimento e prosegui
                - Analizzando i tuoi ragionamenti precedenti {riflessioni_research} individua le sottoricette approvate dal validator che sono strettamente necessarie per realizzarla (non citare gli ingredienti).
                - DEVI SOLO CONSIDERARE COME SOTTORICETTE della ricetta madre quelle presenti in {riflessioni_research}.
                - Per ciascuna sottoricetta individuata, verifica se ESISTE tra i documenti forniti
                    (DB o WEB) uno che la tratti specificamente con ingredienti e procedimento propri.
                    Se NESSUN documento fornito tratta quella sottoricetta, dichiaralo esplicitamente
                    (es. "sottoricetta 'Maionese' necessaria ma non reperita in nessun documento
                    fornito") — questo segnale è OBBLIGATORIO ed evita che il post finale contenga
                    dati inventati per quella sottoricetta.
                
                -verificare la coerenza con i claim esistenti  {claim_text}.
                -Se un claim contraddice la ricetta o le sottoricette, segnalalo nel `think_tool`.
                - Nel `think_tool` DEVI SEMPRE indicare quanti claim hai trovato e se sono coerenti o in conflitto con la ricetta.
                - Se ci claim contraddittori, dichiara: "contraddizione trovata. La ricetta non è valida!" e assegna SCORE 0 ai documenti contradittori.
                - se non ci sono claim contraddittori, dichiara: "nessuna contraddizione trovata. La ricetta è valida!".
        
                ▶ NEL CAMPO 'valutazione_opzioni' (ESEGUI FASE 3 ADESSO):
                - Prendi i documenti RIMASTI (quelli non ancora eletti o scartati).  -
                - SE il documento è una sottoricetta della ricetta madre: ASSEGNA Score 1 al migliore e scarta gli altri assegandoli Score 0(sono duplicati inferiori).
                - SE il documento è irrilevante,fuori tema o non serve: assegna SCORE 0.
        
        
                ▶ NEL CAMPO 'decisione_finale' (VERDETTO FINALE):
                - Scrivi la LISTA FISICA ED ESATTA delle tue valutazioni.
                - L'elenco DEVE contenere ESATTAMENTE {totale_doc} righe (una per ogni documento che ti ho fornito).
                - Formato obbligatorio per ogni riga:
                ID_DOC [TITOLO]: Score X - Motivo: ...
                - REGOLA CRITICA SU ID_DOC (controllata a valle da codice automatico):
                ID_DOC è ESATTAMENTE il numero N che compare nell'etichetta con cui il documento
                ti è stato presentato sopra: "WEB_DOC_N" oppure "DB_DOC_N". L'indicizzazione è
                0-based: il primo documento di ciascuna sezione è WEB_DOC_0 (non WEB_DOC_1) o
                DB_DOC_0 (non DB_DOC_1). NON rinumerare i documenti a partire da 1 come faresti
                elencandoli a un umano: copia il numero N letteralmente dall'etichetta originale
                che hai visto in "DOCUMENTI DA VALUTARE". Esempio corretto: se il secondo
                documento web ti è stato presentato come "WEB_DOC_1", la riga corrispondente
                deve iniziare con "WEB_DOC_1", non "WEB_DOC_2".
                - PRIMA di scrivere "STATO: FINITO", ricontrolla ogni riga: il prefisso (DB_DOC
                oppure WEB_DOC) deve corrispondere ESATTAMENTE alla sezione originale ([DB LOCALE]
                o [WEB]) da cui hai preso quel documento. Un'etichetta sbagliata qui fa perdere
                il documento a valle, anche se lo hai approvato con Score 1.
                - scrivi alla fine del CAMPO 'decisione_finale' " STATO: FINITO validazione completata "
                
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
    else:
        print("sono il validator do il verdetto")
        # ── SECONDA PASSATA: leggi la riflessione dai messaggi e dai il verdetto ──
        # Il tool_node ha già stampato il ragionamento, ci basta il contenuto
        riflessione_testo = ""
        for msg in reversed(messaggi):
            if isinstance(msg, ToolMessage) and msg.name == "think_tool":
                riflessione_testo = msg.content.replace(
                    "Riflessione registrata con successo: ", ""
                ).strip()
                break
        print("sono dentro ")
        prompt_verdetto = f"""
            Sulla base esclusiva di questa tua analisi logica appena effettuata:
            {riflessione_testo}
    
            Produci il verdetto strutturato finale per il topic '{topic}'.
    
            ATTENZIONE (CRITICO):
            1. Devi estrarre e mappare TUTTI i documenti a cui hai assegnato SCORE 1 nella tua analisi.
            2. Usa gli STESSI IDENTICI ID_DOC (i numeri esatti) che hai scritto nell'analisi. È severamente vietato inventare ID non presenti nel testo.
            Ricorda: l'indicizzazione è 0-based (il primo documento di ogni sezione è l'indice 0, non 1). Se l'analisi menziona "WEB_DOC_1", il campo id nel
            JSON deve essere l'intero 1, non 2 — copia il numero letteralmente, non rinumerare
            3. Puoi (e DEVI) approvare UNO O PIÙ documenti (sia DB che Web) se uno rappresenta la Ricetta Madre e gli altri sono le Sottoricette necessarie.
            4. Smista ogni ID nella lista corretta in base al SUO PREFISSO nel testo: un ID scritto
            come "DB_DOC_n" va SEMPRE in ranking_db, un ID scritto come "WEB_DOC_n" va SEMPRE
            in ranking_web. Non dedurre la lista da altro (autorevolezza, contenuto, ecc.),
            usa esclusivamente il prefisso testuale.
            5. Nel DB locale erano presenti {len(dati_db_locale)} documenti, nel WEB {len(dati_web_grezzi)}.
            Se nell'analisi non compare nessun "DB_DOC_x", ranking_db deve restare vuoto (non
            inventare voci). Lo stesso vale simmetricamente per ranking_web.
    
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

            web_ordinato = sorted(
                esito.ranking_web, key=lambda x: x.score, reverse=True
            )

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
    dati_kg = state.get("kg_documents", [])
    feedback = state.get("human_feedback", "")
    post_precedente = state.get("post_draft", "")

    lista_tracce = state.get("reasoning_trace", [])
    traces_formattate = (
        "\n".join([f"- {t}" for t in lista_tracce])
        if lista_tracce
        else "Nessuna traccia di ragionamento precedente registrata."
    )

    testi_approvati = "\n\n".join(dati_db_locale + dati_web + dati_kg)
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
        try:
            claims_semantici = kg_client.get_claim_pertinenti(topic)
            if claims_semantici:
                blocco_claim_semantici = (
                    "=== CLAIM SEMANTICAMENTE PERTINENTI DAL KNOWLEDGE GRAPH ===\n"
                )
                for i, c in enumerate(claims_semantici, 1):
                    blocco_claim_semantici += f"{i}. [{c['topic_correlato']}] (sim: {c['similarità']})\n   \"{c['claim']}\"\n"
                blocco_claim_semantici += "\n"
            else:
                blocco_claim_semantici = ""
        except Exception as e:
            blocco_claim_semantici = ""
            print(f"[WRITER] Errore recupero claim semantici: {e}")
        num_claim_sem = len(claims_semantici) if claims_semantici else 0
        print(
            f"[WRITER] KG: stile={'ok' if style else 'vuoto'} | "
            f" | claim_post={len(claim_correlati)} | correlati={len(topic_correlati)}"
            f"claim_semantici={num_claim_sem} trovati"
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



     - 🚨 REGOLA ANTI-RICORSIONE FONDAMENTALE: La ricetta principale del post (ovvero '{topic}') è la Ricetta Madre. NON PUÒ MAI ESSERE INSERITA TRA LE SOTTORICETTE. I suoi ingredienti e i suoi passaggi devono andare rigorosamente e unicamente in 'ingredienti_diretti' e 'preparazione'.
     - 🚨 è VIETATO CREARE delle sottoricette non incluse nelle precedenti tracce di ragionamento: :{traces_formattate}.
     - 🚨 Gli ingredienti delle sottoricette non DEVONO MAI COMPARIRE all'interno di ingredienti diretti pure se citati dalla fonte della ricetta madre stessa in quanto non fanno parte della praprazione della sottoricetta.
        ========================================================================
        CONTESTO EDITORIALE DAL KNOWLEDGE GRAPH
        ========================================================================

        [STILE DEL BLOG — rispetta queste linee guida in ogni scelta redazionale]
        {blocco_stile}

        [CLAIM GIÀ PUBBLICATI — richiamali esplicitamente se pertinenti, evita di ripeterli]
        {blocco_claim}
        [CLAIM SEMANTICAMENTE PERTINENTI — usali per collegamenti o approfondimenti]
        {blocco_claim_semantici if blocco_claim_semantici else "Nessun claim semantico aggiuntivo."}
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
            - Se sono presenti CLAIM SEMANTICAMENTE PERTINENTI, usali come ulteriore
                fonte di ispirazione per creare collegamenti con altri post del blog.
            - Se esiste un topic correlato (ingredienti in comune), puoi aggiungere
                un breve rimando alla fine dell'introduzione (es. "Se ami questo piatto,
                potresti trovare interessante anche la nostra ricetta di X").
            - Rispetta SEMPRE il tono, il registro e la lunghezza target indicati
                nelle linee guida stilistiche

        ### 2. RIGORE STRUTTURALE (Coesione Ricetta → Sottoricette)

                L'output deve rispecchiare ESATTAMENTE l'albero delle dipendenze definito nelle tracce di ragionamento:
                {traces_formattate}.

                #### Regole obbligatorie

                . **Rispettare il ragionamento**
                - Se il ragionamento identifica un elemento come SOTTORICETTA, esso DEVE essere rappresentato esclusivamente all'interno di `sotto_ricette`.
                - È SEVERAMENTE VIETATO creare sottoricette che non compaiono esplicitamente nel ragionamento, anche se la fonte della ricetta le descrive.

                . **Estrazione completa della sottoricetta**
                - Per ogni sottoricetta presente nel ragionamento, estrai dalla relativa fonte tutti gli ingredienti e il procedimento disponibili.
                - La sottoricetta deve essere rappresentata integralmente nella sezione `sotto_ricette`.

                🚨. **Ingredienti della Ricetta Madre**
                - Gli ingredienti diretti della Ricetta Madre devono contenere ESCLUSIVAMENTE gli ingredienti elementari utilizzati direttamente nella preparazione della Ricetta Madre.
                - Gli ingredienti appartenenti a una sottoricetta NON devono comparire negli ingredienti diretti della Ricetta Madre, anche se ti tratta di ingredienti concettualmente simili che 
                  compaiono nella fonte originale della ricetta madre

                🚨. **Eliminazione delle duplicazioni**
                - Se un ingrediente è stato classificato come sottoricetta, rimuovi dagli ingredienti diretti della Ricetta Madre:
                    - il nome della sottoricetta (es. "Besciamella", "Ragù", "Crema pasticcera");
                    - tutti gli ingredienti utilizzati per prepararla (es. latte, burro, farina nel caso della besciamella), anche se sono riportati nella fonte della Ricetta Madre.
                - Gli ingredienti di una sottoricetta devono comparire UNA SOLA VOLTA, esclusivamente all'interno della relativa sottoricetta. 

                . **Divieto di rappresentazioni ibride**
                - È SEVERAMENTE VIETATO lasciare una sottoricetta come semplice ingrediente della Ricetta Madre.
                - È SEVERAMENTE VIETATO rappresentare contemporaneamente una sottoricetta sia negli ingredienti diretti della Ricetta Madre sia nella sezione `sotto_ricette`.

                . **Coerenza strutturale**
                - Ogni ingrediente deve appartenere ad un solo livello dell'albero:
                    - oppure alla Ricetta Madre;
                    - oppure a una Sottoricetta;
                    - mai ad entrambi contemporaneamente.

        3. ANCHORING METRICO-TESTUALE:
            - Estrai dosi, pesi, ingredienti e passaggi ESCLUSIVAMENTE dai testi approvati forniti. Non inventare ingredienti né arrotondare le dosi.
            - Se un ingrediente manca di quantità nel testo, scrivi "q.b." o "quantità non specificata".

        4. REGOLA ANTI-SOPRASTRUTTURE:
            - Non creare oggetti SottoRicetta per ingredienti pronti o topping che
                non subiscono trasformazione (es. parmigiano da spolverare, prosciutto
                pronto): questi vanno negli ingredienti diretti.
            - Non creare MAI una SottoRicetta per bagne, sciroppi, succhi di frutta, o qualsiasi liquido dolce preparato al momento (es. "Bagna di Fragole", "Bagna al Caffè", "Sciroppo di Zucchero"). 
                Questi vanno descritti direttamente nei passaggi della preparazione principale

        5. DETTAGLIO DELLA PREPARAZIONE:
            - Elenca i passaggi in modo esteso, analitico e sequenziale.
            - Ogni passaggio è una stringa chiara e indipendente nella lista.
            - Non riassumere procedimenti complessi in un solo paragrafo.
        
        6. FONTI
            -cita OBBLIGATORIAMENTE TUTTE le fonti UTILIZZATE per la stesura del post nella sezione FONTI

        ========================================================================
        FONTI E RAGIONAMENTI
        ========================================================================

        [TRACCE DI RAGIONAMENTO (LOG DEL GRAPH STATE)]
        {traces_formattate}

        [TESTI DELLE FONTI APPROVATE (RICETTA MADRE + SOTTORICETTE)]
        
        {testi_approvati}
        [MODIFICHE RICHIESTE DALL'UTENTE]
        
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
                "human_feedback": None,
                "rag_documents": [None],  # grazie a replace_or_add
                "web_documents": [None],
                "kg_documents": [None],
                "approved_db_documents": [],
                "approved_web_documents": [],
                "recipe_draft": None,
            },
            goto="research",
        )

    elif feedback.startswith("MODIFICA:"):
        testo_feedback = feedback.replace("MODIFICA:", "").strip()
        print(f"[HITL] Modifica richiesta: '{testo_feedback}'")
        return Command(update={"human_feedback": testo_feedback}, goto="writer")

    elif feedback == "SCARTA":
        print("[HITL]  Post scartato dall'utente.")
        input_utente = state.get("input_utente", "")
        piano = state.get("piano_editoriale")
        indice_corrente = state.get("indice_post_corrente", 0)

        messaggi_da_cancellare = [
            RemoveMessage(id=m.id) for m in state.get("messages", [])
        ]

        if input_utente == "PIANIFICAZIONE_AUTOMATICA":

            nuovo_indice = indice_corrente + 1
            if piano and nuovo_indice < len(piano.sequenza_post):
                prossimo_topic = piano.sequenza_post[nuovo_indice].topic
                print(f"[HITL] Prossimo topic: {prossimo_topic}")

                return Command(
                    update={
                        # Stessi reset  di kg_update_node
                        "messages": messaggi_da_cancellare,
                        "indice_post_corrente": nuovo_indice,
                        "topic_corrente": prossimo_topic,
                        "rag_documents": [None],
                        "web_documents": [None],
                        "kg_documents": [None],
                        "approved_web_documents": [],
                        "approved_db_documents": [],
                        "is_valid": None,
                        "recipe_draft": None,
                        "post_draft": None,
                        "human_feedback": None,
                        "is_rigenera": False,
                    },
                    goto="research",
                )

            else:

                print("[HITL] Nessun altro topic nel piano: fine esecuzione.")
                return Command(
                    update={
                        "messages": messaggi_da_cancellare,
                        "indice_post_corrente": nuovo_indice,
                        "human_feedback": None,
                        "post_draft": None,
                    },
                    goto=END,
                )
        else:
            print("[HITL] Modalità singolo topic: fine esecuzione.")
            return Command(
                update={
                    "messages": messaggi_da_cancellare,
                    "human_feedback": None,
                    "post_draft": None,
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
    categoria = draft.categoria

    # ── CADENZA EDITORIALE SIMULATA ──────────────────────────────
    # Il post N del piano viene "pubblicato" N * GIORNI_TRA_POST giorni
    # dopo l'inizio del piano editoriale, anche se generato nella stessa sessione
    ultima_data = kg_client.get_ultima_data_pubblicazione()

    if ultima_data:
        data_pubblicazione = ultima_data + timedelta(days=GIORNI_TRA_POST)
    else:
        data_pubblicazione = datetime.now()

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
            "procedimento": sub.procedimento,
            "categoria": sub.categoria,
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
 Sei un tassonomista culinario. Devi identificare la **RADICE MADRE** di una ricetta, cioè il piatto base da cui quella specifica versione deriva.

**REGOLE RIGIDE (in ordine di priorità):**

⚠️REGOLA ANTI-RICORSIONE: La ricetta principale (il topic '{topic_finale}') NON può mai essere considerata una sottoricetta di se stessa. 
 Quando analizzi gli ingredienti, escludi a priori il nome della ricetta che stai preparando.

1. **Modifiche dietetiche o di cottura**  
   Se il nome contiene una parola che indica una MODIFICA DIETETICA o DI COTTURA riconoscibile (es. "light", "vegana", "senza glutine", "senza uova", "senza lattosio", "integrale", "al forno", "fritta"), elimina quella parola.  
   *Esempi:*  
   - "Carbonara light" → "Carbonara"  
   - "Tiramisù vegano" → "Tiramisù"  
   - "Melanzane alla parmigiana al forno" → "Melanzane alla parmigiana"  

2. **Varianti di gusto / condimento su una base neutra**  
   Se il nome è composto da un PIATTO BASE NEUTRO (che può esistere da solo) + un complemento di gusto/condimento (es. "alla fragola", "al pistacchio" ), la radice è il piatto base.  
   *Attenzione:* la base deve essere un piatto autonomo, riconoscibile e comunemente declinato in più gusti. Rientrano in questa categoria: **Pizza,  Gelato, Tiramisù, Panna cotta, Crostata, Torta (semplice), Focaccia, ecc.**  
   *Esempi:*  
   - "Tiramisù alla fragola" → "Tiramisù"   
   - "Pizza capricciosa" → "Pizza"  
   - "Gelato al pistacchio" → "Gelato"  

3. **Piatti autonomi non derivati**  
   Se il nome è un piatto completo e specifico che non rappresenta una variante di gusto di una base neutra, ma ha una propria identità (es. "Pasta alla carbonara", "Bruschette al pomodoro", "Caponata", "Pollo al limone", "Lasagne alla bolognese"), la radice è il nome stesso.  
   *Esempi:*  
   - "Pasta alla carbonara" → "Pasta alla carbonara" (non si riduce a "Pasta")  
   - "Bruschette al pomodoro" → "Bruschette al pomodoro"  
   - "Caponata" → "Caponata"  
   
4 **ingredienti come nomi dei piati**
    Se il nome del piatto è un ingrediente enfatizzato con qualche aggettivo allora in quel caso non eliminare l'aggetivo e considera come ricetta madre il nome completo del piatto
    *Esempi:*
    - "Spinaci Sfiziosi"-> "Spinaci Sfiziosi"
    - "melanzare arrabbiate-> "melanzare arrabbiate"
    - "patate povere"-> "patate povere"


5. **Casi ambigui: verifica della base**  
   Dopo aver applicato una rimozione (regola 1 o 2), controlla se il nome risultante è un piatto noto e autonomo. Se non lo è, **mantieni il nome originale** (non forzare l'estrazione).  
   *Esempio:* "Pasta al pesto" non diventa "Pasta" perché "Pasta" da sola non è un piatto specifico; la radice rimane "Pasta al pesto".  


Ora analizza: **'{topic_finale}'**

Rispondi **SOLO** con il nome della radice madre, senza commenti o spiegazioni.
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
            procedimento_principale="\n".join(draft.preparazione),
            categoria=categoria,
            data_pubblicazione=data_pubblicazione,
        )

        print("[NEO4J] Salvataggio completato.")

    except Exception as e:
        print(f"[ERRORE NEO4J] Si è verificato un problema durante il salvataggio: {e}")

    nuovo_indice = indice_corrente + 1
    piano = state.get("piano_editoriale")
    messaggi_da_cancellare = [RemoveMessage(id=m.id) for m in state.get("messages", [])]
    # Se ci sono altri post, prepara il prossimo topic
    if piano and nuovo_indice < len(piano.sequenza_post):
        prossimo_topic = piano.sequenza_post[nuovo_indice].topic
        print(f"[KG UPDATE] Prossimo topic: {prossimo_topic}")
        return {
            "messages": messaggi_da_cancellare,
            "indice_post_corrente": nuovo_indice,
            "topic_corrente": prossimo_topic,
            # Resettiamo i campi transitori
            "rag_documents": [None],
            "web_documents": [None],
            "kg_documents": [None],
            "approved_web_documents": [],
            "approved_db_documents": [],
            "is_valid": None,
            "recipe_draft": None,
            "post_draft": None,
            "human_feedback": "",
            "is_rigenera": False,
        }
    else:
        return {"indice_post_corrente": nuovo_indice}
