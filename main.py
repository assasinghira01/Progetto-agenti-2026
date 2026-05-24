from rag.vector_db import inizializza_vector_db
from graph.workflow import app

if __name__=="__main__":

    inizializza_vector_db()
    
    print("=== INIZIO TEST AGENTE MODULARE K-RAG ===")
    input_test = "Ciao, oggi il mio pubblico mi chiede di parlare di caponata. Puoi aiutarmi?"

    risultato_finale = app.invoke({"input_utente": input_test})

    print("\n==============================================")
    print(" OUTPUT FINALE GENERATO DALL'LLM")
    print("==============================================")
    print(risultato_finale["post_draft"])