import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from knowledge_graph.neo4j_manager import CucinaKnowledgeGraph

# Inizializziamo il client fuori dalla funzione per non riaprire la connessione ogni volta
# 1. Carichiamo le chiavi dal file .env
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
kg_client = CucinaKnowledgeGraph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

@tool
def controlla_storico_post(topic: str) -> str:
    """
    Usa questo strumento per interrogare il Knowledge Graph e scoprire se 
    nel blog è già stato pubblicato un articolo su un determinato piatto 
    o su una sua variante (es. Caponata Catanese vs Palermitana).
    
    Da usare SEMPRE all'inizio della pianificazione per evitare ripetizioni.
    
    Args:
        topic: Il nome del piatto da controllare.
    """
    try:
        # Questa è la funzione Cypher che naviga le relazioni (che hai in neo4j_manager.py)
        # Supponiamo che restituisca una stringa formattata con lo storico
        risultato = kg_client.controlla_cronologia_post(topic)
        
        if risultato:
            return f"STORICO TROVATO:\n{risultato}"
        return f"Nessun post precedente trovato per '{topic}' o le sue varianti. Puoi procedere."
    except Exception as e:
        return f"Errore di connessione a Neo4j: {str(e)}"