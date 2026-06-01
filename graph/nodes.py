from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from config import llm, llm_con_tools
from graph.state import Blog_Cucina
from knowledge_graph.neo4j_manager import kg_client


class ValidationResult(BaseModel):
    is_valid: bool = Field(
        description="True se le fonti sono coerenti e contengono una ricetta logica e fattibile, False se la richiesta contiene assurdità o mancano dati fondamentali."
    )
    reasoning: str = Field(
        description="Spiegazione dettagliata del perché hai accettato o rifiutato le fonti."
    )


# Schema di output
class TopicExtraction(BaseModel):
    topic: str = Field(
        description="Il nome del piatto o ingrediente principale richiesto."
    )


# Prepariamo l'LLM strutturato da usare dentro il planner_node

llm_structured = llm.with_structured_output(TopicExtraction)


def planner_node(state: Blog_Cucina):
    print("\n--- [NODO 1: PLANNER (LLM)] ---")
    input_utente = state["input_utente"]

    # 1. Estraiamo il topic principale
    risultato = llm_structured.invoke(
        [HumanMessage(content=f"Estrai il topic: {input_utente}")]
    )
    topic_estratto = risultato.topic.capitalize()

    print(f" Topic identificato: {topic_estratto}")

    # 2. Scriviamo il topic nello stato e passiamo la palla al nodo di ricerca
    return {"topic_corrente": topic_estratto}


def krag_research_node(state: Blog_Cucina):
    print("\n--- [NODO 2: RICERCA MCP (Agente Autonomo)] ---")
    topic = state["topic_corrente"]
    messaggi = state.get("messages", [])

    istruzioni_mcp = SystemMessage(content=f"""
    Sei l'agente investigatore del blog di cucina, nel nsotro blog parleremo di ricette, sagre eventi ecc. Il topic ATTUALE su cui devi lavorare è: '{topic}'.
    
    REGOLE DI COMPORTAMENTO (rispettale rigorosamente in ordine):
    1. **Prima azione assoluta verifica che il post non sia gia stato pubblicato**: chiama 'controlla_storico_post' su '{topic}'.
    
    2. Analizza la risposta del Knowledge Graph:
       - Se inizia con "BLOCCATO": significa che il post è già stato pubblicato di recente.
       
         → Chiama 'krag_suggerisci_varianti'  se parliamo di ricette per ottenere gli ingredienti base e 3 varianti creative.
         → Dopo aver ricevuto le varianti, chiama IMMEDIATAMENTE 'chiedi_variante' passando le 3 opzioni all'utente.
         → FERMATI. Non invocare altri tool.
        
       
       - Se inizia con "OK": significa che il post è nuovo. Devi raccogliere le informazioni seguendo questo esatto flusso di ragionamento (Thought -> Action):
         
            Sai che il nostro database locale contiene solo ricette quindi non troverai nulla su sagre, eventi ecc. Effetua la ricerca su DB LOCALE usando 'cerca_ricetta_nel_db' solo se parliamo di ricette. Nel caso in cui
            L'utente richieda varianti o modifiche delle ricette devi cercare informazioni sul web utilizzando "esegui_ricerca_web" DEVI estrarre gli INGREDIENTI, le DOSI e il PROCEDIMENTO.
            
           Nel caso in cui l'utente non parli di ricette non chiamare 'cerca_ricetta_nel_db' e vai direttamente con 'esegui_ricerca_web' per raccogliere informazioni di contesto sul '{topic}' in questione.
         
         
    
    3. Se ricevi "ERRORE" o "KRAG_ERRORE", segnala immediatamente il problema all'utente e chiedi istruzioni.
    """)

    if not messaggi:
        messaggi_da_inviare = [
            istruzioni_mcp,
            HumanMessage(content=f"Inizia la ricerca per: {topic}"),
        ]
    else:
        messaggi_da_inviare = messaggi + [istruzioni_mcp]

    risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)
    return {"messages": [risposta_llm]}


def validator_node(state: Blog_Cucina):
    print("\n--- [NODO 3: VALIDATORE (Fact-Checking Incrociato)] ---")
    topic = state["topic_corrente"]

    # pescando i messaggi di tipo "tool" dalla cronologia dell'Agente.
    dati_db_locale = []
    dati_web = []

    # Cicliamo sui messaggi per estrarre il contenuto in base al tool che l'ha generato
    for m in state["messages"]:
        if hasattr(m, "type") and m.type == "tool":
            nome_tool = getattr(m, "name", "")

            if nome_tool == "cerca_ricetta_nel_db":
                if "Nessuna ricetta" not in m.content:
                    dati_db_locale.append(m.content)

            elif nome_tool == "esegui_ricerca_web":
                dati_web.append(m.content)

    testo_db = (
        "\n".join(dati_db_locale)
        if dati_db_locale
        else "NESSUNA RICETTA TROVATA NEL DB LOCALE"
    )
    testo_web = "\n".join(dati_web) if dati_web else "NESSUN DATO COLLATERALE DAL WEB"

    prompt = f"""Analizza la fattibilità editoriale per il piatto: '{topic}'.
    
    DATI RACCOLTI DAGLI STRUMENTI (DB Locale e Web):
     
  
    === FONTE DI VERITÀ INTERNA (DB LOCALE) ===
    {testo_db}
    
    === INFORMAZIONI DI CONTESTO (RICERCA WEB) ===
    {testo_web}
    
  
    COMPITO E CRITERI DI VALUTAZIONE:
    1. **Senso Gastronomico (Coerenza)**: La richiesta ha senso logico dal punto di vista culinario? Blocca immediatamente ricette assurde, accostamenti improponibili o disgustosi (es. "Tiramisù al merluzzo", "Carbonara con la Nutella").
    2. **Sufficienza dei Dati**: Verifica semplicemente se, unendo il DB Locale e la Ricerca Web, abbiamo abbastanza informazioni (ingredienti, dosi o procedimenti minimi) per poter scrivere un articolo sensato su questo piatto.
    3. **Esito**: 
       - Imposta la validazione su True se il piatto è gastronomicamente valido e ci sono dati sufficienti per parlarne.
       - Imposta la validazione su False se il piatto è un'assurdità culinaria o se non è stato trovato assolutamente nulla in nessuna delle due fonti.
       
    """

    llm_validator = llm.with_structured_output(ValidationResult)
    esito = llm_validator.invoke([HumanMessage(content=prompt)])

    print(f" Esito Validazione: {esito.is_valid}")
    print(f" Motivazione dell'LLM: {esito.reasoning}")
    return {"is_valid": esito.is_valid}


def writer_node(state: Blog_Cucina):
    print("\n--- [NODO 4: WRITER (Sintesi e Grounding)] ---")
    topic = state["topic_corrente"]

    dati_db_locale = []
    dati_web = []

    for m in state["messages"]:
        if hasattr(m, "type") and m.type == "tool":
            if getattr(m, "name", "") == "cerca_ricetta_nel_db":
                if "Nessuna ricetta" not in m.content:
                    dati_db_locale.append(m.content)
            elif getattr(m, "name", "") == "esegui_ricerca_web":
                dati_web.append(m.content)

    ha_db_locale = len(dati_db_locale) > 0
    testo_db = "\n".join(dati_db_locale) if ha_db_locale else "NESSUN DATO IN LOCALE"
    testo_web = "\n".join(dati_web) if dati_web else "NESSUN DATO DAL WEB"

    feedback = state.get("human_feedback")
    istruzione_correzione = (
        f"\nATTENZIONE - RICHIESTA DEL CAPO REDATTORE: {feedback}\nAdatta la ricetta seguendo questa istruzione."
        if feedback
        else ""
    )

    regola_gerarchia_fonti = (
        f"""
Se nel {testo_db} trovi la ricetta CORRETTA (non varianti o modifiche) DEVI usare ESCLUSIVAMENTE gli ingredienti, le dosi (es. 1l di latte, 100g di burro, 100g di farina per la Besciamella) e il procedimento descritti lì. 
Se nel {testo_db} non trovi la modifica o variante richiesta dall'utente, allora DEVI  usare ESCLUSIVAMENTE i dati del {testo_web} per gli ingredienti, le dosi e il procedimento."""
        if ha_db_locale
        else f"""
Il DB Locale non ha restituito risultati. Usa i dati della {testo_web} per estrarre ingredienti, dosi esatte e procedimento."""
    )

    prompt_sistema = f"""Sei un food blogger professionista. Scrivi un post su: {topic} con breve introduzione (max 30 parole).

    === GERARCHIA DELLE FONTI DI VERITÀ (RISPETTA RIGIDAMENTE): ===
    {regola_gerarchia_fonti}
    DEVI SPECIFICARE LE DOSI ESATTE DEGLI INGREDIENTI presi dalla fonte che stai utilizzando (es. 100g di farina, 1l di latte) E UN RIASSUNTO BREVE DEL PROCEDIMENTO(MAX 100 PAROLE , NON UTILIZZARE ELENCHI PUNTATI PER IL PROCEDIMENTO MA FAI UN RIASSUNTO):  SE E SOLO SE SONO PRESENTI NELLE FONTI FORNITE.
    NON DEVI MAI USARE GLI INGREDIENTI CONTENUTI NEL DB LOCALE SE LA RICETTA TROVATA NON CORRISPONDE ALLA VARIANTE RICHIESTA DALL'UTENTE. IN QUEL CASO DEVI USARE SOLO I DATI DELLA RICERCA WEB.
    Cita le fonti utilizzate a fine articolo (se usi il DB locale, indica come fonte la struttura interna o la fonte specificata nel payload del DB. altrimenti cita la fonte del web. In entrambi i casi metti il link). {istruzione_correzione}=== DATI COMPILATI DAL DB LOCALE ==={testo_db}=== DATI COMPILATI DALLA RICERCA WEB ==={testo_web}"""

    risposta_llm = llm.invoke(prompt_sistema)
    return {"post_draft": risposta_llm.content}


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
    topic_finale = state["topic_corrente"]

    # Estraiamo il concetto puro (senza le estensioni della variante) dall'input originario
    risultato_originale = llm_structured.invoke(
        [
            HumanMessage(
                content=f"Estrai il piatto principale (no varianti) da: {state['input_utente']}"
            )
        ]
    )
    topic_originale = risultato_originale.topic

    try:
        # Passiamo entrambe le entità per mappare la gerarchia corretta
        kg_client.salva_post(topic_originale, topic_finale)
        print(
            f"STORICO AGGIORNATO: Il post su '{topic_finale}' è stato salvato nel Grafo di Neo4j!"
        )
    except Exception as e:
        print(f"Errore durante il salvataggio su Neo4j: {str(e)}")

    return {}


def aggiorna_topic_node(state: Blog_Cucina):
    """
    Nodo che estrae la scelta dell'utente dal tool chiedi_variante
    e aggiorna topic_corrente per la ricerca successiva.
    """
    ultimo_messaggio = state["messages"][-1]
    nuovo_topic = None

    if (
        hasattr(ultimo_messaggio, "content")
        and "SCELTA_UTENTE|" in ultimo_messaggio.content
    ):
        scelta = ultimo_messaggio.content.split("|")[-1].strip()
        # Possiamo anche pulire la scelta (es. "1. Caponata al forno" -> "Caponata al forno")
        if scelta[0].isdigit() and ". " in scelta:
            nuovo_topic = scelta.split(". ", 1)[-1]
        else:
            nuovo_topic = scelta
        print(f"[AGGIORNA TOPIC] Nuovo topic impostato a: '{nuovo_topic}'")
        return {"topic_corrente": nuovo_topic}

    return {}
