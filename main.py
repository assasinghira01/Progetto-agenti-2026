import uuid
import os
from dotenv import load_dotenv

# 1. Caricamento immediato delle variabili d'ambiente (api key, credenziali db)
load_dotenv()

from rag.vector_db import inizializza_vector_db
from graph.workflow import app
from langgraph.types import Command


def main():
    print("====================================================")
    print(" 🍳 AI Food Blogger Copilot - Sistema K-RAG + HITL ")
    print("====================================================")

    # Inizializziamo o carichiamo il database vettoriale locale
    inizializza_vector_db()

    richiesta = input(
        "\nDi cosa vorresti parlare oggi? (es. 'Scrivi un post sulla caponata'):\n> "
    )
    if not richiesta.strip():
        print("Input vuoto. Terminazione del programma.")
        return

    # Creiamo un ID di sessione univoco (Thread) per mantenere attiva la memoria del grafo
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n🚀 Avvio del flusso di lavoro dell'Agente...\n")

    # --- PRIMO AVVIO DEL GRAFO ---
    # Girerà finché non incontra la FINE o un punto di interruzione (interrupt)
    for event in app.stream({"input_utente": richiesta}, config):
        for nome_nodo, var_modificate in event.items():
            print(f"[LOG GRAFO] Il nodo '{nome_nodo}' ha terminato l'esecuzione.")
            # Se il nodo ha estratto il topic, lo mostriamo subito come feedback
            if "topic_corrente" in var_modificate:
                print(f"Topic identificato: {var_modificate['topic_corrente']}")
        pass  # I log dettagliati di avanzamento sono stampati internamente dai nodi

    # --- LOOP DINAMICO DI GESTIONE INTERRUPT ---
    # Gestisce n interruzioni consecutive (scelta variante -> validazione -> revisione finale)
    while True:
        stato_grafo = app.get_state(config)

        # Se non ci sono nodi successivi nella coda (.next), il grafo ha concluso la sua esecuzione
        if not stato_grafo.next:
            break

        print("\n[SISTEMA IN PAUSA] Rilevato punto di interruzione Human-in-the-Loop.")

        # Estraggiamo i dati correnti memorizzati nello stato del thread
        valori_stato = stato_grafo.values
        bozza_articolo = valori_stato.get("post_draft")
        messaggi = valori_stato.get("messages", [])

        # STRATEGIA DI DISCRIMINAZIONE DELL'INTERRUPT:
        # Se nello stato NON è ancora presente la bozza dell'articolo, siamo fermi alla scelta della variante!
        if not bozza_articolo:
            ultimo_messaggio_agente = (
                messaggi[-1].content if messaggi else "Nessuna opzione visualizzabile."
            )
            print("\n----------------------------------------------------")
            print("L'AGENTE CHIEDE DI SCEGLIERE UNA VARIANTE STORICA:")
            print("----------------------------------------------------")
            print(ultimo_messaggio_agente)

            feedback_utente = input(
                "\nQuale variante decidi di sviluppare? (Digita la tua scelta):\n> "
            )

        # Se la bozza è presente, significa che abbiamo superato la ricerca e siamo alla revisione finale del testo!
        else:
            print("\n----------------------------------------------------")
            print("BOZZA FINALE GENERATA DALL'LLM PER IL TUO BLOG:")
            print("----------------------------------------------------")
            print(bozza_articolo)
            print("----------------------------------------------------")

            feedback_utente = input(
                "\nInserisci il tuo verdetto (es. 'Approvo', oppure indica le modifiche):\n> "
            )

        print("\nRipresa dell'esecuzione con l'input fornito...\n")

        # Risvegliamo il grafo inviando il comando di sblocco (resume) con il feedback dell'utente
        for event in app.stream(Command(resume=feedback_utente), config):
            for nome_nodo, var_modificate in event.items():
                print(f"[LOG GRAFO] Ripresa attività. Eseguito nodo: '{nome_nodo}'")
            pass

    # --- CONCLUSIONE DEL FLUSSO ---
    # Recuperiamo lo stato finale per stampare l'esito definitivo dell'elaborazione
    stato_finale = app.get_state(config)
    post_finale = stato_finale.values.get("post_draft")

    if post_finale:
        print("\n==============================================")
        print("PROCESSO COMPLETATO CON SUCCESSO!")
        print("==============================================")
        print(
            "Il post è stato ufficialmente validato, approvato e memorizzato nel grafo Neo4j."
        )
        print("Ottimo lavoro!")
    else:
        print("\n==============================================")
        print(" FLUSSO TERMINATO SENZA PUBBLICAZIONE")
        print("==============================================")
        print("Il processo è stato interrotto  del Validatore per incoerenza dei dati.")

    print(" resta sempre collegato con noi !!.")


if __name__ == "__main__":
    main()
