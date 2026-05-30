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

    # Se è la prima volta che entriamo nel nodo, diamo all'agente le istruzioni
    if not messaggi:
        sys_msg = SystemMessage(content=f"""Sei l'investigatore del blog culinario.
        Il tuo compito è raccogliere TUTTE le informazioni necessarie sul piatto: '{topic}'.
        
        REGOLE:
        La primissima azione che DEVI compiere è chiamare lo strumento 'controlla_storico_post' 
        per verificare se abbiamo già parlato di '{topic}'. Non usare altri strumenti prima di questo.
        
        DOPO aver letto la risposta dello storico di Neo4j:
        1. Se il piatto è GIA' STATO PUBBLICATO: FERMATI. NON usare altri tool (non cercare sul web o nel DB). 
           Rispondi all'utente proponendo 3 varianti creative e alternative e chiedigli esplicitamente 
           quale preferisce esplorare, inserendo un punto di domanda.
        2. Se il piatto è NUOVO (o se l'utente ha già scelto la variante al turno precedente): 
           Usa contemporaneamente gli altri tuoi tools 'cerca_ricetta_nel_db' e 'esegui_ricerca_web'.
           
        Usa i tool finché non hai raccolto tutto quello che serve per le regole sopra.
        """)
        messaggi = [sys_msg, HumanMessage(content=f"Inizia la ricerca per: {topic}")]

    # L'LLM decide autonomamente se chiamare un tool o se ha finito
    risposta_llm = llm_con_tools.invoke(messaggi)

    # Restituiamo il messaggio. Se l'LLM ha deciso di usare un tool,
    # questo messaggio conterrà una richiesta speciale (tool_calls)
    return {"messages": [risposta_llm]}


def human_variant_node(state: Blog_Cucina):
    print("\n L'Agente attende la scelta della variante...")

    # L'interruzione avviene qui, in modo sicuro all'interno del nodo
    scelta_variante = interrupt("Quale variante scegli? ")

    print(f"Variante selezionata dall'utente: '{scelta_variante}'")

    # Ritorniamo i dati aggiornati: ora LangGraph li salverà permanentemente nello Stato!
    return {
        "messages": [
            HumanMessage(
                content=f"Scelgo questa variante: {scelta_variante}. Procedi con la ricerca su questa variante."
            )
        ],
        "topic_corrente": scelta_variante,  # Sovrascriviamo il topic con la variante!
    }


def validator_node(state: Blog_Cucina):
    print("\n--- [NODO 3: VALIDATORE (Fact-Checking Incrociato)] ---")
    topic = state["topic_corrente"]

    # NOVITÀ: Estraiamo TUTTO quello che i Tool (DB, Web, Neo4j) hanno trovato
    # pescando i messaggi di tipo "tool" dalla cronologia dell'Agente.
    dati_raccolti = "\n\n".join(
        [
            m.content
            for m in state["messages"]
            if hasattr(m, "type") and m.type == "tool"
        ]
    )

    prompt = f"""Analizza la fattibilità editoriale per il piatto: '{topic}'.
    
    DATI RACCOLTI DAGLI STRUMENTI (DB Locale e Web):
    {dati_raccolti}
    
    COMPITO:
    1. Verifica se la richiesta dell'utente ha senso logico e gastronomico (es. abbinamenti assurdi di ingredienti).
    2. Controlla se i dati raccolti contengono le informazioni minime per scrivere una ricetta (ingredienti e passaggi base).
    
    Se la richiesta è un'assurdità o mancano i dati fondamentali, imposta is_valid=False. Altrimenti imposta True.
    """

    llm_validator = llm.with_structured_output(ValidationResult)
    esito = llm_validator.invoke([HumanMessage(content=prompt)])

    print(f" Esito Validazione: {esito.is_valid}")
    print(f" Motivazione dell'LLM: {esito.reasoning}")
    return {"is_valid": esito.is_valid}


def writer_node(state: Blog_Cucina):
    print("\n--- [NODO 4: WRITER (Sintesi e Grounding)] ---")
    topic = state["topic_corrente"]

    # Anche qui, estraiamo i dati di ricerca dalla memoria
    dati_raccolti = "\n\n".join(
        [
            m.content
            for m in state["messages"]
            if hasattr(m, "type") and m.type == "tool"
        ]
    )

    # Recuperiamo l'eventuale feedback umano (se il nodo viene ri-eseguito dopo che tu hai chiesto modifiche!)
    feedback = state.get("human_feedback")
    istruzione_correzione = (
        f"\nATTENZIONE - RICHIESTA DEL CAPO REDATTORE: {feedback}\nAdatta la ricetta seguendo questa istruzione."
        if feedback
        else ""
    )

    prompt_sistema = f""" Sei un food blogger professionista. Scrivi un post di massimo 150 parole su: {topic}.
    
    REGOLE DI GROUNDING:
    1. Usa le dosi esatte presenti nei dati locali per la scheda tecnica del piatto (non inventare).
    2. Usa le curiosità web per inserire un'introduzione discorsiva su trend o varianti.
    3. Cita le fonti a fine articolo. {istruzione_correzione}
    
    DATI RACCOLTI DAGLI STRUMENTI:
    {dati_raccolti}
    """

    risposta_llm = llm.invoke(prompt_sistema)
    return {"post_draft": risposta_llm.content}


def human_review_node(state: Blog_Cucina):
    print("\n--- [NODO 5: HUMAN-IN-THE-LOOP (Approvazione)] ---")
    bozza = state.get("post_draft", "")

    print("\n================ BOZZA DEL POST ================\n")
    print(bozza)
    print("\n================================================\n")

    #  LA MAGIA: Il grafo si congela qui e invia questo messaggio all'interfaccia (o al terminale)
    # L'esecuzione si ferma finché non inserisci un valore per "feedback"
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
