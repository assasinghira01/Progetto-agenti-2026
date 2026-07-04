import os
import json
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

    DIVIETO ASSOLUTO: Non usare MAI query brevi o parole generiche (es. vietato "Ricetta della maionese").
    Devi generare un documento ipotetico denso di contesto (HyDE) per abbattere le distanze vettoriali.

    ISTRUZIONI DI GENERAZIONE DELLA QUERY (K-RAG ibrido):
    La tua query deve essere un paragrafo discorsivo generato seguendo rigorosamente UNA di queste due casistiche:

    CASO A - K-RAG PURO (Se hai estratto dati dal Knowledge Graph in precedenza):
    Se possiedi già una lista di ingredienti o claim recuperati dal grafo, la tua query DEVE
    contenere il nome del piatto e utilizzare ESCLUSIVAMENTE quegli ingredienti storici e claim recuperati.
    Il procedimento tecnico deve riguardare  SOLO i claim recuperati.

    CASO B - HyDE FALLBACK (Se il Knowledge Graph era vuoto o non hai dati pregressi):
    Se il piatto è inedito, devi allucinare tu l'intero documento. La tua query DEVE contenere:
    1. Il nome del piatto.
    2. Una lista coerente di ingredienti dedotti dalla tua conoscenza interna (es. se cerchi maionese: uova, olio, limone).
    3. Un mini-procedimento tecnico discorsivo.

    ESEMPIO DI QUERY CORRETTA E RIGOROSA CHE DEVI EMULARE:
     "Nome della ricetta completa". Ingredienti: "ingredienti estratti dal kg". Procedimento: "claim estratti dal kg "

    ESEMPIO DI QUERY ERRATA DA SCARTARE:
    "Ricetta della maionese con ingredienti e procedimento."
    """
    if vector_store is None:
        return "Errore di sistema: Il database locale non è inizializzato."

    try:
        SOGLIA_DISTANZA = 0.25
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
            return json.dumps(documenti_recuperati, ensure_ascii=False)

        return "Nessuna ricetta ufficiale pertinente trovata nel database locale."

    except Exception as e:
        print(f"[RAG ERRORE]: {str(e)}")
        return f"Errore interno del database vettoriale: {str(e)}"
