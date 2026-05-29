from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Specifichiamo la stessa cartella usata in vector_db.py
DB_DIR = "./chroma_db_cucina"

@tool
def cerca_ricetta_nel_db(query: str) -> str:
    """
    Usa ESCLUSIVAMENTE questo strumento per cercare le ricette ufficiali, 
    le dosi esatte, le grammature e i procedimenti base dei piatti.
    Questa è la tua fonte di verità principale per gli ingredienti.
    
    Args:
        query: Il nome del piatto o gli ingredienti da cercare (es. "Caponata").
    """
    try:
        # Inizializziamo l'embedding direttamente dentro il tool
        embeddings_locali = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings_locali 
        )
        
        risultati = vector_store.similarity_search(query, k=3)
        
        if risultati:
            testo_risultati = "\n\n".join([res.metadata.get("payload", "") for res in risultati])
            return testo_risultati
        return "Nessuna ricetta trovata nel database locale per questa query."
    except Exception as e:
        return f"Errore di connessione al database vettoriale: {str(e)}"