from langchain_chroma import Chroma
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil

# Percorsi ASSOLUTI basati sulla posizione di questo file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_DIR = os.path.join(ROOT_DIR, "chroma_db_cucina")
CARTELLA_RICETTE = os.path.join(ROOT_DIR, "ricettario_md")


def popola_database_rag():
    print(f"1. Scansione della cartella '{CARTELLA_RICETTE}' in corso...")

    loader = DirectoryLoader(
        CARTELLA_RICETTE,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},  # Fix encoding
    )
    documenti = loader.load()

    if not documenti:
        print("Nessun file Markdown trovato.")
        return

    print(f" -> Trovate {len(documenti)} ricette base.")
    print("2. Inizializzazione del modello di Embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("3. Creazione del Vector Store (ChromaDB)...")

    # Cancella il DB vecchio per evitare duplicati e embedding corrotti
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    vector_store = Chroma.from_documents(
        documents=documenti,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    print("\n Database vettoriale RAG popolato con successo!")
    print(f"   Salvato in: {DB_DIR}")
