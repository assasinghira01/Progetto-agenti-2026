import os
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_DIR = os.path.join(ROOT_DIR, "chroma_db_cucina")

print("[SISTEMA] Inizializzazione motore di ricerca semantica RAG...")
embeddings_locali = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

if os.path.exists(DB_DIR):
    vector_store = Chroma(
        persist_directory=DB_DIR, embedding_function=embeddings_locali
    )
else:
    vector_store = None
    print(
        "[ATTENZIONE] Nessun database Chroma trovato. Esegui crea_vector_db.py prima."
    )


@tool
def cerca_ricetta_nel_db(query: str) -> str:
    """
    Usa ESCLUSIVAMENTE questo strumento per cercare preparazioni base,
    ricette ufficiali del blog e liste di ingredienti nel tuo database RAG locale.
    Questa è la FONTE DI VERITÀ ASSOLUTA per le ricette interne.
    """
    if vector_store is None:
        return "Errore di sistema: Il database locale non è inizializzato."

    try:
        SOGLIA_DISTANZA = 1.0
        risultati_con_distanza = vector_store.similarity_search_with_score(query, k=3)

        documenti_recuperati = []

        for doc, dist in risultati_con_distanza:
            nome_file = doc.metadata.get("source", "File Sconosciuto")

            # Bug 1: il filtro era SEPARATO dal loop che costruisce i risultati,
            # quindi scartava ma poi aggiungeva tutto lo stesso nel secondo loop
            if dist > SOGLIA_DISTANZA:
                print(f"[RAG] Scartato '{nome_file}' (distanza {dist:.4f} > soglia)")
                continue

            testo_chunk = doc.page_content
            blocco_formattato = (
                f"=== FONTE AUTOREVOLE LOCALE: {nome_file} ===\n"
                f"{testo_chunk}\n"
                f"==================="
            )
            documenti_recuperati.append(blocco_formattato)
            print(f"[RAG DEBUG] Trovata: '{nome_file}' (distanza: {dist:.4f})")

        if documenti_recuperati:
            return "\n\n".join(documenti_recuperati)

        return "Nessuna ricetta ufficiale pertinente trovata nel database locale."

    except Exception as e:
        print(f"[RAG ERRORE]: {str(e)}")
        return f"Errore interno del database vettoriale: {str(e)}"
