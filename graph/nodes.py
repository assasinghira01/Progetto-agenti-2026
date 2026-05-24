from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import Blog_Cucina
from config import llm, llm_structured
from knowledge_graph.neo4j_manager import kg_client
from rag.vector_db import cerca_ricetta_nel_db

def planner_node(state: Blog_Cucina):
    print("\n--- [NODO 1: PLANNER (LLM)] ---")
    risultato = llm_structured.invoke([
        SystemMessage(content="Sei un assistente editoriale. Estrai l'argomento culinario."),
        HumanMessage(content=state["input_utente"])
    ])
    topic_estratto = risultato.topic
    print(f"LLM ha estratto il topic puro: '{topic_estratto}'")
    return {"topic_corrente": topic_estratto}

def krag_research_node(state: Blog_Cucina):
    topic_corrente = state["topic_corrente"]
    print(f"\n--- [NODO 2: K-RAG] Avvio ricerca per: '{topic_corrente}' ---")
    
    termini_espansi = kg_client.espandi_query_per_topic(topic_corrente)
    if termini_espansi:
        stringa_espansa = " ".join(termini_espansi)
        query_finale = f"{topic_corrente} {stringa_espansa}"
        print(f"Grafo consultato. Query espansa: '{query_finale}'")
    else:
        query_finale = topic_corrente
        print(" Nessun dato nel Grafo. Uso query standard.")
        
    documento_estratto = cerca_ricetta_nel_db(query=query_finale)
    print(f"\nRAG HA RECUPERATO:\n{documento_estratto[:300]}")  
    
    return {"kg_context": termini_espansi, "rag_documents": [documento_estratto]}

def writer_node(state: Blog_Cucina):
    print("\n--- [NODO 3: WRITER (LLM Generativo)] ---")
    topic = state["topic_corrente"] 
    documento = state["rag_documents"][0]
    
    prompt_sistema = f"""
    Sei un food blogger. Scrivi un breve post introduttivo su: {topic}.
    Elenca ingredienti con precise grammature e intolleranze.
    Regola fondamentale: Usa ESCLUSIVAMENTE le informazioni estratte.
    Cita sempre fonte e link. Non inventare nulla.
    
    DATI DATABASE:
    {documento}
    """
    print(" GPT-4o-mini sta scrivendo l'articolo...")
    risposta_llm = llm.invoke(prompt_sistema)
    return {"post_draft": risposta_llm.content}