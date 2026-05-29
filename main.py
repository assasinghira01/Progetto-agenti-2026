import uuid
import os
from dotenv import load_dotenv

from graph.workflow import app
from langgraph.types import Command

load_dotenv()

def main():
    print(" Benvenuto nell'AI Food Blogger Copilot!")
    print("------------------------------------------")
    richiesta = input("Di cosa vorresti parlare oggi? (es. 'Scrivi un post sulla caponata'):\n> ")
    
   # 1. Creiamo un ID di sessione.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("\n Avvio del ragionamento dell'Agente...\n")
    # 2. Avviamo il grafo. Girerà in automatico passando da Planner, Research, Validatore e Writer.
    for event in app.stream({"input_utente": richiesta}, config):
        # Stampa a schermo i nomi dei nodi mano a mano che vengono eseguiti
        for nome_nodo, stato_ritornato in event.items():
            pass # I print dettagliati sono già dentro i nostri file nodes.py
            
    # 3. INTERRUZIONE UMANA (Il grafo è in pausa)
    stato_grafo = app.get_state(config)
    
    # Se .next contiene qualcosa, significa che il grafo non è finito (è bloccato al nodo 'human_review')
    if stato_grafo.next:
        print("\n Il sistema è in pausa. In attesa di approvazione umana!")
        
        # Chiediamo all'utente cosa ne pensa della bozza stampata a schermo
        feedback_utente = input("\n Il tuo verdetto (es. 'Approvo', oppure 'Mettici meno sale'):\n> ")
        
        print("\n Ripresa dell'esecuzione...\n")
        
        # 4. RIPRESA DEL GRAFO. Usiamo 'Command' per rimettere in moto l'agente 
        # iniettando la risposta dell'utente direttamente nel nodo in pausa.
        for event in app.stream(Command(resume=feedback_utente), config):
             for nome_nodo, stato_ritornato in event.items():
                 pass

    print("\n Processo completato! Il post è stato salvato.")
    print(" Puoi vedere l'albero decisionale completo sulla tua dashboard di LangSmith.")

if __name__ == "__main__":
    main()