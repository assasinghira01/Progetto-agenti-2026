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
kg_client = CucinaKnowledgeGraph(
    uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD
)


@tool
def get_ultimi_post():
    """
    Torna gli ultimi post pubblicati sul blog per evitare ripetizioni.
    Restituisce l'elenco dei titoli dei post recenti.
    """
    # Recuperiamo la lista dei post (restituisce una lista di dict)
    risultati = kg_client.get_ultimi_post_pubblicati(limite=30)

    if not risultati:
        return "Nessun post precedente è stato pubblicato nel blog. Il blog è vuoto."

    # Estraiamo i titoli e formattiamoli in una stringa pulita
    elenco_titoli = [post["titolo"] for post in risultati]
    stringa_post = ", ".join(elenco_titoli)

    return f"Ultimi post trovati nel blog: {stringa_post}"


@tool
def controlla_storico_post(topic: str) -> str:
    """
    Verifica se il piatto è già stato pubblicato.
    Restituisce "BLOCCATO" + dettagli se duplicato, altrimenti "OK".
    """
    risultato = kg_client.controlla_cronologia_post(topic)
    if risultato:
        return f"BLOCCATO|Il piatto '{topic}' è già stato pubblicato come '{risultato['titolo_post']}'"
    return f"OK|Nessun post precedente per '{topic}'"


@tool
def krag_suggerisci_varianti(topic: str) -> str:
    """
    Usa il Knowledge Graph per ottenere gli ingredienti base e suggerire varianti creative.
    Da chiamare solo dopo che controlla_storico_post ha restituito BLOCCATO.
    """
    ingredienti_base = kg_client.espandi_query_per_krag(topic)
    if not ingredienti_base:
        return "KRAG_ERRORE|Nessun ingrediente trovato nel grafo per generare varianti."

    # Costruisci un messaggio ricco per l'agente
    varianti = f"INGREDIENTI_BASE: {', '.join(ingredienti_base)}\n"
    varianti += "SUGGERISCI_QUESTE_3_VARIANTI:\n"
    varianti += "1. Variante light (riduci grassi, aggiungi erbe aromatiche)\n"
    varianti += "2. Variante ricca (aggiungi formaggi o salumi tipici)\n"
    varianti += "3. Variante di stagione (usa ingredienti diversi in base al periodo)\n"
    return varianti
