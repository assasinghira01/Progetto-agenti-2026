from rag.vector_db import inizializza_vector_db
from graph.workflow import app

if __name__ == "__main__":
    # 1. Controllo e inizializzazione automatica del database locale.
    # Grazie al controllo interno, se la cartella esiste già non farà perdere tempo.
    inizializza_vector_db()
    
    print("\n=== INIZIO TEST AGENTE MODULARE K-RAG ===")
    
    
    input_test = "Ciao, oggi il mio pubblico mi chiede di parlare di un tiramisu alle fragole. Puoi aiutarmi?"
    
 

    print(f"Input inviato all'agente: '{input_test}'")
    print("Esecuzione del grafo in corso... (Controlla i log dei nodi qui sotto)")
    
    # Avviamo LangGraph passando lo stato iniziale richiesto
    risultato_finale = app.invoke({"input_utente": input_test})

    print("\n==============================================")
    print(" VERDETTO E OUTPUT DEL SISTEMA")
    print("==============================================")
    
    # Usiamo .get() per evitare crash se il validatore ha interrotto il flusso
    bozza_articolo = risultato_finale.get("post_draft")
    
    if bozza_articolo:
        print("✅ ARTICOLO APPROVATO E GENERATO CON SUCCESSO:\n")
        print(bozza_articolo)
    else:
        print("🛑 FLUSSO INTERROTTO DAL GUARDAIL DI SICUREZZA!")
        print("Il Nodo Validatore ha interrotto l'esecuzione prima della scrittura.")
        print("Motivazione: La richiesta conteneva un'incoerenza logica o i dati nel RAG erano insufficienti.")