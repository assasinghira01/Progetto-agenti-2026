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

        # Con similarity_search_with_score prendiamo le distanze grezze (L2 o Cosine)
        # Non lancia eccezioni sui punteggi negativi o fuori range [0,1]
        risultati_con_distanza = vector_store.similarity_search_with_score(query, k=5)

        print(f"\n[DEBUG DISTANZE PER {query}]:")
        risultati_filtrati = []
        for doc, dist in risultati_con_distanza:
            payload = doc.metadata.get("payload", "")
            # Estrai il titolo (assumendo formato "Ricetta: Nome ...")
            titolo = ""
            if "Ricetta:" in payload:
                titolo = payload.split("Ricetta:")[1].split("\n")[0].strip().lower()
            print(f" -> Piatto: {titolo[:30]}... | Distanza: {dist}")

            # Filtro per similarità semantica + corrispondenza testuale
            # Se la distanza è accettabile (es. < 1.0) E il titolo contiene la query (o viceversa)
            if dist < 1.0 and (query.lower() in titolo or titolo in query.lower()):
                risultati_filtrati.append(doc)
            else:
                print(f"    Scartato perché non contiene '{query}'")

        if risultati_filtrati:
            testo_risultati = "\n\n".join(
                [res.metadata.get("payload", "") for res in risultati_filtrati]
            )
            return testo_risultati

        return (
            "Nessuna ricetta sufficientemente pertinente trovata nel database locale."
        )

    except Exception as e:
        # Se fallisce di nuovo, questo print ti dirà l'errore esatto nel terminale
        print(f"[DEBUG ERRORE CHROMA]: {str(e)}")
        return f"Errore di connessione al database vettoriale: {str(e)}"
