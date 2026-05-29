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
        risultato = kg_client.controlla_cronologia_post(topic)
        
        if risultato:
            # Creiamo il messaggio severo per l'LLM
            messaggio = (f" STOP! Il piatto '{topic}' è GIÀ PRESENTE nel database "
                         f"(Titolo esistente: '{risultato['titolo_post']}'). "
                         f"REQUISITO DI SISTEMA: È severamente vietato scrivere un post identico. "
                         f"DEVI obbligatoriamente cercare e scrivere una VARIANTE (es. '{topic} al forno').")
            
            # STAMPIAMO a schermo per te (così lo vedi nel terminale!)
            print(f"\n [TOOL NEO4J] {messaggio}")
            
            # RITORNIAMO all'LLM
            return messaggio
            
        # Se è un piatto nuovo:
        messaggio_ok = f"Nessun post precedente trovato per '{topic}'. Puoi procedere con la ricetta classica."
        
        # Stampiamo a schermo per te e poi diamo il via libera all'LLM
        print(f"\n [TOOL NEO4J]  {messaggio_ok}")
        return messaggio_ok
        
    except Exception as e:
        errore = f"Errore di connessione a Neo4j: {str(e)}"
        print(f"\n [TOOL NEO4J]  {errore}")
        return errore