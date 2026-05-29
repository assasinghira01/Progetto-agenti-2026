from langchain_chroma import Chroma
from datasets import load_dataset
from langchain_core.documents import Document

import os
DB_DIR = "./chroma_db_cucina"

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def inizializza_vector_db():
    
    if os.path.exists(DB_DIR):
        print(f"[ChromaDB] Database già presente in '{DB_DIR}'. Salto la vettorizzazione.")
        return None
    
    print("Scarico il dataset da Hugging Face...")
    dataset = load_dataset("KanuSaru/italian-recipes", split="train")
    campione_dev = dataset.select(range(10))
    
    print("Trasformo le ricette in 'Documenti' per LangChain...")
    documenti_langchain = []
    
    for ricetta in campione_dev:
        titolo_ricetta = str(ricetta.get('Nome', 'Ricetta Senza Nome'))
        categoria = str(ricetta.get('Categoria', 'Senza categoria'))
        ingredienti = str(ricetta.get('Ingredienti', ''))
        procedimento = str(ricetta.get('Steps', ''))
        porzioni = str(ricetta.get('Persone/Pezzi', 'N/A'))
        
        link = str(ricetta.get('Link', '')).lower()
        fonte = link.split('//')[-1].split('/')[0].replace('www.', '').replace('ricette.', '').split('.')[0].capitalize()
        
        testo_da_vettorizzare = (
            f"Titolo piatto: {titolo_ricetta}. "
            f"Categoria: {categoria}. "
            f"Ingredienti principali: {ingredienti}."
        )
        
        testo_ricetta = (
            f"Ricetta: {titolo_ricetta}\n"
            f"Fonte: {fonte}\n"
            f"Porzioni: {porzioni}\n"
            f"Categoria: {categoria}\n"
            f"Ingredienti: {ingredienti}\n"
            f"Procedimento: {procedimento}\n"
            f"Link:{link}" 
        )
        
        doc = Document(
            page_content=testo_da_vettorizzare,
            metadata={"payload": testo_ricetta}
        )
        documenti_langchain.append(doc)
        
    print("Vettorizzo e salvo nel database Chroma...")
    vector_store = Chroma.from_documents(
        documents=documenti_langchain,
        embedding=EMBEDDINGS,  
        persist_directory=DB_DIR
    )
    
    print("Vector DB pronto!")
    return vector_store

