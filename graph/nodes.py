from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import Blog_Cucina
from config import llm, llm_structured
from knowledge_graph.neo4j_manager import kg_client
from rag.vector_db import cerca_ricetta_nel_db
from pydantic import BaseModel, Field
from tools.search_tool import esegui_ricerca_web



class ValidationResult(BaseModel):
    is_valid: bool = Field(description="True se le fonti sono coerenti e contengono una ricetta logica e fattibile, False se la richiesta contiene assurdità o mancano dati fondamentali.")
    reasoning: str = Field(description="Spiegazione dettagliata del perché hai accettato o rifiutato le fonti.")



def planner_node(state: Blog_Cucina):
    print("\n--- [NODO 1: PLANNER (LLM)] ---")
    risultato = llm_structured.invoke([
        SystemMessage(content="Sei un assistente editoriale. Estrai l'argomento culinario."),
        HumanMessage(content=state["input_utente"])
    ])
    topic_estratto = risultato.topic
    duplicato_trovato = kg_client.controlla_cronologia_post(topic_estratto)
    
    if duplicato_trovato:
        print(f"Neo4j rileva una sovrapposizione storica per: '{topic_estratto}' (Già trattato in '{duplicato_trovato['titolo_post']}')!")
        
        risultato_alternativo = llm_structured.invoke([
            SystemMessage(content=f"""
            Il blog ha già trattato il piatto '{topic_estratto}'. 
            Analizza la richiesta originale dell'utente e proponi un piatto alternativo strettamente correlato 
            o una variante regionale (es. se la richiesta era sulla Carbonara, proponi la Gricia o l'Amatriciana).
            """),
            HumanMessage(content=state["input_utente"]) 
        ])
        topic_estratto = risultato_alternativo.topic
        print(f"Strategia editoriale variata. Nuovo Topic alternativo: '{topic_estratto}'")
    else:
        print(f"Topic approvato: '{topic_estratto}' (Nuovo argomento per il blog)")
        
    return {"topic_corrente": topic_estratto}



def krag_research_node(state: Blog_Cucina):
    topic_corrente = state["topic_corrente"]
    print(f"\n--- [NODO 2: K-RAG] Avvio ricerca per: '{topic_corrente}' ---")
    
    termini_espansi = kg_client.espandi_query_per_krag(topic_corrente)
    query_locale = f"{topic_corrente} {' '.join(termini_espansi)}" if termini_espansi else topic_corrente
    
    print(f"🔍 Interrogazione ChromaDB...")
    documento_locale = cerca_ricetta_nel_db(query=query_locale)
    
    print(f"🌐 Interrogazione Tavily Search...")
    documento_web = esegui_ricerca_web(query=f"{topic_corrente} ricetta tradizionale passaggi")
    
    return {
        "kg_context": termini_espansi,
        "rag_documents": [documento_locale],
        "web_documents": [documento_web]
    }
    
    
def validator_node(state: Blog_Cucina):
    print("\n--- [NODO 3: VALIDATORE (Fact-Checking Incrociato)] ---")
    topic = state["topic_corrente"] 
    doc_rag = state["rag_documents"][0]
    doc_web = state["web_documents"][0]
    
    prompt = f"""Analizza la fattibilità editoriale per il piatto: '{topic}'.
    
    FONTE CERTIFICATA LOCALE:
    {doc_rag}
    
    FONTE ESTERNA WEB:
    {doc_web}
    
    COMPITO:
    1. Verifica se la richiesta dell'utente ha senso logico e gastronomico (es. abbinamenti assurdi come pesce e dolci cremosi vanno bocciati).
    2. Controlla se la fonte locale contiene i dati minimi (ingredienti e passaggi).
    
    Se la richiesta è un'assurdità o mancano i dati fondamentali, imposta is_valid=False. Altrimenti imposta True.
    """
    
    llm_validator = llm.with_structured_output(ValidationResult)
    esito = llm_validator.invoke([HumanMessage(content=prompt)])
    
    print(f"🕵️ Esito Validazione: {esito.is_valid}")
    print(f"📝 Motivazione dell'LLM: {esito.reasoning}")
    return {"is_valid": esito.is_valid}



def writer_node(state: Blog_Cucina):
    print("\n--- [NODO 4: WRITER (Sintesi e Grounding)] ---")
    topic = state["topic_corrente"] 
    doc_rag = state["rag_documents"][0]
    doc_web = state["web_documents"][0]
    
    prompt_sistema = f""" Sei un food blogger professionista. Scrivi un post di massimo 150 parole su: {topic}.
    
    REGOLE DI GROUNDING:
    1. Usa le dosi esatte della fonte piu adatta per la scheda tecnica del piatto (non fare un mix).
    2. Usa la FONTE WEB per inserire un'introduzione discorsiva su trend o varianti.
    3. Non inventare dati non presenti nelle fonti. Cita i link utilizzati.
    
    FONTE LOCALE:
    {doc_rag}
    
    FONTE WEB:
    {doc_web}
    """
    risposta_llm = llm.invoke(prompt_sistema)
    return {"post_draft": risposta_llm.content}



