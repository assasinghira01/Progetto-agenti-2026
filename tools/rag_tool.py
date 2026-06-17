import os
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_DIR = os.path.join(ROOT_DIR, "chroma_db_cucina")

print("[SISTEMA] Inizializzazione motore di ricerca semantica RAG...")
embeddings_locali = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

vector_store = None


def inizializza_vector_store():
    """Carica il vector store dal DB. Va chiamata dopo popola_database_rag()."""
    global vector_store
    if os.path.exists(DB_DIR):
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings_locali,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print("[RAG] Vector store caricato correttamente.")
    else:
        vector_store = None
        print("[RAG] Nessun database Chroma trovato.")


@tool
def cerca_ricetta_nel_db(query: str) -> str:
    # tecnia Hyde per il RAG, vogliamo volutamente che LLM allucini per favorire il recupero dei dati !
    """
    [AZIONE OBBLIGATORIA]
    Cerca una ricetta o preparazione base nel database vettoriale locale.

    DIVIETO ASSOLUTO: Non usare MAI query brevi o parole generiche. È severamente
    vietato usare query pigre come "Ricetta della maionese con ingredienti".

    ISTRUZIONI HyDE (Hypothetical Document Embeddings):
    Per abbattere le distanze vettoriali, PRIMA di invocare questo tool devi generare una
    "Ricetta Ipotetica" e passarla come parametro 'query'. NON avere alcuna paura di sbagliare
    dosi, ingredienti o tecniche: il tuo unico scopo è generare massa semantica per il database.

    La tua stringa 'query' DEVE obbligatoriamente contenere queste 3 cose:
    1. Nome del piatto.
    2. Lista esplicita e reale degli ingredienti (deduci tu i principali, es. uova, farina, burro...).
    3. Un mini-procedimento discorsivo (spiega l'azione tecnica: frullare, infornare, mantecare...).

    ESEMPIO DI QUERY CORRETTA E RIGOROSA CHE DEVI EMULARE:
    "Ricetta completa per la maionese. Ingredienti: tuorli d'uovo, olio di semi, succo di limone, sale. Procedimento: Mettere i tuorli in una ciotola, aggiungere il limone e frullare versando l'olio a filo lentamente fino a montare l'emulsione."

    ESEMPIO DI QUERY ERRATA DA SCARTARE:
    "Ricetta della maionese con ingredienti e procedimento."
    """
    if vector_store is None:
        return "Errore di sistema: Il database locale non è inizializzato."

    try:
        SOGLIA_DISTANZA = 0.2
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
