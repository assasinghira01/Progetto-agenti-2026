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
    print(" AI Food Blogger Copilot - Sistema K-RAG + HITL ")
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

    print("\nAvvio del flusso di lavoro dell'Agente...\n")

    # --- PRIMO AVVIO DEL GRAFO ---
    for event in app.stream({"input_utente": richiesta}, config):
        for nome_nodo, var_modificate in event.items():
            if nome_nodo != "__root__":
                print(f"[LOG GRAFO] Il nodo '{nome_nodo}' ha terminato l'esecuzione.")

                # Controllo difensivo per evitare il crash se var_modificate è None o vuoto
                if var_modificate is not None and "topic_corrente" in var_modificate:
                    print(
                        f" -> [UPDATE] Topic corrente impostato a: {var_modificate['topic_corrente']}"
                    )

    # --- LOOP DINAMICO DI GESTIONE INTERRUPT ---
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
        # Scenario A: Se NON è ancora presente la bozza, siamo fermi alla scelta della variante!
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

            # Sblocchiamo inviando la scelta e forziamo l'aggiornamento di human_feedback
            comando_sblocco = Command(
                resume=feedback_utente, update={"human_feedback": feedback_utente}
            )

        # Scenario B: Se la bozza è presente, siamo alla revisione finale del testo del post!
        else:
            print("\n----------------------------------------------------")
            print("BOZZA FINALE GENERATA DALL'LLM PER IL TUO BLOG:")
            print("----------------------------------------------------")
            print(bozza_articolo)
            print("----------------------------------------------------")

            feedback_utente = input(
                "\nInserisci il tuo verdetto (es. 'Approvo', oppure indica le modifiche):\n> "
            )

            # Sblocchiamo passando il feedback che verrà consumato da human_review_node
            comando_sblocco = Command(
                resume=feedback_utente, update={"human_feedback": feedback_utente}
            )

        print("\nRipresa dell'esecuzione con l'input fornito...\n")

        # Riattiviamo lo stream passando il comando configurato
        for event in app.stream(comando_sblocco, config):
            for nome_nodo, var_modificate in event.items():
                if nome_nodo != "__root__":
                    print(
                        f"[LOG GRAFO] Ripresa attività. Completato nodo: '{nome_nodo}'"
                    )

                    # Controllo difensivo analogo anche in fase di ripresa dello stream
                    if (
                        var_modificate is not None
                        and "topic_corrente" in var_modificate
                    ):
                        print(
                            f" -> [UPDATE] Il topic è cambiato in: '{var_modificate['topic_corrente']}'"
                        )

    # --- CONCLUSIONE DEL FLUSSO ---
    stato_finale = app.get_state(config)
    post_finale = stato_finale.values.get("post_draft")
    is_valid = stato_finale.values.get("is_valid")

    if post_finale and is_valid != False:
        print("\n==============================================")
        print(" PROCESSO COMPLETATO CON SUCCESSO! 🎉")
        print("==============================================")
        print(
            "Il post è stato ufficialmente validato, approvato e memorizzato nel grafo Neo4j."
        )
        print("Ottimo lavoro!")
    else:
        print("\n==============================================")
        print(" FLUSSO TERMINATO SENZA PUBBLICAZIONE 🛑")
        print("==============================================")
        print(
            "Il processo è stato interrotto dal Validatore per incoerenza dei dati o mancanza di fonti."
        )

    print("\nResta sempre collegato con noi!!")


if __name__ == "__main__":
    main()
