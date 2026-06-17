import uuid

from dotenv import load_dotenv

# 1. Caricamento immediato delle variabili d'ambiente (api key, credenziali db)
load_dotenv()

from rag.vector_db import popola_database_rag
from tools.rag_tool import inizializza_vector_store
from graph.workflow import app
from langgraph.types import Command


def main():
    print("====================================================")
    print(" AI Food Blogger Copilot - Sistema K-RAG + HITL ")
    print("====================================================")

    # Inizializziamo o carichiamo il database vettoriale locale
    popola_database_rag()
    inizializza_vector_store()

    while True:
        print("\n=== MENU PRINCIPALE ===")
        print("1. Pianificazione automatica (piano editoriale)")
        print("2. Scrivi un post specifico")
        print("3. Esci")

        scelta = input("Scegli un'opzione (1-3): ").strip()

        if scelta == "1":
            richiesta = "PIANIFICAZIONE_AUTOMATICA"
            break
        elif scelta == "2":
            richiesta = input("Scrivi un post (es. 'tiramisù'):\n> ").strip()
            if not richiesta:
                print("Topic non valido, riprova.")
                continue
            break
        elif scelta == "3":
            print("Uscita dal programma.")
            return
        else:
            print("Scelta non valida, riprova.")
            continue

    # Creiamo un ID di sessione univoco (Thread) per mantenere attiva la memoria
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\nAvvio del flusso di lavoro dell'Agente...\n")

    # --- AVVIO DEL GRAFO ---
    for event in app.stream({"input_utente": richiesta}, config):
        for nome_nodo, var_modificate in event.items():
            if nome_nodo != "__root__":
                print(f"[LOG GRAFO] Il nodo '{nome_nodo}' ha terminato l'esecuzione.")

    # --- LOOP DINAMICO DI GESTIONE INTERRUPT ---

    while True:
        stato_grafo = app.get_state(config)

        # Se non ci sono nodi successivi (.next), il grafo ha terminato
        if not stato_grafo.next:
            break

        print("\n[SISTEMA IN PAUSA] Rilevato punto di interruzione Human-in-the-Loop.")

        prossimo_nodo = stato_grafo.next[0]  # Il primo nodo in attesa
        comando = None

        if prossimo_nodo == "human_review_planner":
            piano = stato_grafo.values.get("piano_editoriale")
            if not piano:
                print("Errore: nessun piano trovato.")
                break

            print("\n--- PIANO EDITORIALE ---")
            piano = stato_grafo.values.get("piano_editoriale")
            for i, post in enumerate(piano.sequenza_post):
                print(f"[{i+1}] Ricetta: {post.topic} \n categoria:({post.categoria})")
                print(f"giustificazione: {post.giustificazione}\n")

            scelta = input(
                "\n 1) APPROVA PIANO" "\n2) RIGENERA PIANO:\n" "3) MODIFICA PIANO:\n"
            ).strip()

            # Usiamo il Command per decidere il nodo successivo
            if scelta == "1":
                comando = Command(resume="APPROVA")

            elif scelta == "2":
                comando = Command(resume="RIGENERA")

            elif scelta == "3":
                istruzioni = input(
                    "\n📝 Scrivi le tue istruzioni di modifica (es. 'Sostituisci il dolce con un antipasto'):\n> "
                ).strip()
                comando = Command(resume=f"MODIFICA:{istruzioni}")
            else:
                print("si è verificato un errore")
                continue

        # --------------------------------------HUMAN REVIEW VARIANTE--------------------------------------
        elif prossimo_nodo == "human_review_variante":
            topic_proposto = stato_grafo.values.get("topic_corrente", "Sconosciuto")

            print(
                "\n Attenzione: Il topic originale è già stato trattato in passato.\n"
                f" Il sistema propone la variante: '{topic_proposto}' "
                "basata sugli ingredienti principali del piatto bloccato.\n"
                "È possibile scegliere una delle seguenti opzioni."
            )
            print("\n--- OPZIONI ---")
            scelta = input("\n 1=APPROVA" " \n 2= RIGENERA" " \n 3= MODIFICA\n").strip()
            if scelta == "1":
                comando = Command(resume="APPROVA")
            elif scelta == "2":
                comando = Command(resume="RIGENERA")
            elif scelta == "3":
                istruzioni = input(
                    "\n📝 Scrivi le tue istruzioni di modifica:\n> "
                ).strip()
                comando = Command(resume=f"MODIFICA:{istruzioni}")
            else:
                print("si è verificato un errore")
                continue
        # --------------------------------------BOZZA POST--------------------------------------
        elif prossimo_nodo == "human_review":
            bozza = stato_grafo.values.get("post_draft")
            if not bozza:
                print("Errore: nessuna bozza trovata.")
                break

            print("\n--- BOZZA POST ---\n", bozza, "\n----------------")
            scelta = input("1=APPROVA  2=MODIFICA  3=RIGENERA: ").strip()

            if scelta == "1":
                comando = Command(resume="APPROVA")
            elif scelta == "2":
                modifica = input("Modifiche: ")
                comando = Command(resume=modifica)
            else:
                comando = Command(resume="RIGENERA")
        else:
            print(f"Nodo {prossimo_nodo} non gestito, esco.")
            break

        print("\nRipresa dell'esecuzione con l'input fornito...\n")
        if comando:
            for event in app.stream(comando, config):
                for nome_nodo, var_modificate in event.items():
                    if nome_nodo != "__root__":
                        print(
                            f"[LOG GRAFO] Ripresa attività. Completato nodo: '{nome_nodo}'"
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
