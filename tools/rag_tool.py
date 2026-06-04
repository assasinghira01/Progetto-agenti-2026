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
    """
    try:
        embeddings_locali = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        vector_store = Chroma(
            persist_directory=DB_DIR, embedding_function=embeddings_locali
        )

        # Prendiamo i 5 documenti più vicini semanticamente
        risultati_con_distanza = vector_store.similarity_search_with_score(query, k=3)

        print(f"\n[DEBUG DISTANZE PER {query}]:")
        documenti_recuperati = []

        for doc, dist in risultati_con_distanza:
            payload = doc.metadata.get("payload", "")

            # Estraiamo il titolo solo per fare un log pulito sul terminale
            titolo = ""
            if "Ricetta:" in payload:
                titolo = payload.split("Ricetta:")[1].split("\n")[0].strip().lower()

            print(f" -> Recuperato: {titolo[:30]}... | Distanza: {dist}")

            documenti_recuperati.append(doc)

        if documenti_recuperati:
            # Uniamo i payload di tutti i documenti trovati
            testo_risultati = "\n\n".join(
                [res.metadata.get("payload", "") for res in documenti_recuperati]
            )
            return testo_risultati

        return "Nessuna ricetta trovata nel database locale."

    except Exception as e:
        # Se fallisce di nuovo, questo print ti dirà l'errore esatto nel terminale
        print(f"[DEBUG ERRORE CHROMA]: {str(e)}")
        return f"Errore di connessione al database vettoriale: {str(e)}"
