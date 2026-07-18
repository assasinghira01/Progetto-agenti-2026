import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from knowledge_graph.neo4j_manager import CucinaKnowledgeGraph

# Inizializziamo il client fuori dalla funzione per non riaprire la connessione ogni volta
# 1. Carichiamo le chiavi dal file .env
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
kg_client = CucinaKnowledgeGraph(
    uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD
)


@tool
def get_ultimi_post():
    """
    Torna gli ultimi post pubblicati sul blog per evitare ripetizioni.
     Restituisce titolo, topic trattato e i claim chiave di ciascun post recente..
    """
    # Recuperiamo la lista dei post (restituisce una lista di dict)
    risultati = kg_client.get_ultimi_post_pubblicati(limite=30)

    if not risultati:
        return "Nessun post precedente è stato pubblicato nel blog. Il blog è vuoto."

    # Estraiamo i dati
    righe = []
    for post in risultati:
        riga = f"- {post['titolo']} (topic: {post['topic_trattato']})"
        claims = post.get("claims", [])
        if claims:
            # Max 2 claim per post: bastano al planner per capire il contenuto
            # già trattato senza gonfiare inutilmente il contesto
            claim_str = "; ".join(claims)
            riga += f"\n  Claim chiave: {claim_str}"
        righe.append(riga)

    return "Ultimi post pubblicati nel blog:\n" + "\n".join(righe)


@tool
def controlla_storico_post(topic: str) -> str:
    """
    Verifica se il piatto è già stato pubblicato.
    Restituisce "BLOCCATO" + dettagli se duplicato, altrimenti "OK".
    """
    risultato = kg_client.controlla_cronologia_post(topic)
    if risultato:
        return f"BLOCCATO|Il piatto '{topic}' è già stato pubblicato come '{risultato['titolo_post']}'"
    return f"OK|Nessun post precedente per '{topic}'"


@tool
def get_ingredienti(nome_ricetta: str) -> str:
    """
    Interroga il Knowledge Graph per estrarre gli ingredienti di una ricetta.

    """
    ingredienti = kg_client.espandi_query_per_krag(nome_ricetta)

    if not ingredienti:
        print(f"[TOOL] Nessun dettaglio trovato nel grafo per '{nome_ricetta}'.")
        return f"Nessun dettaglio/ingrediente trovato nel grafo per '{nome_ricetta}'. Sii creativo."

    # Restituisce una stringa chiara per l'LLM
    elenco = ", ".join(ingredienti)
    print(f"[TOOL] Ingredienti principali per '{nome_ricetta}': {elenco}")
    return f"DATI ESTRATTI PER '{nome_ricetta}': Gli ingredienti principali sono [{elenco}]."


@tool
def get_claim_pertinenti(topic: str) -> str:
    """
    Recupera i claim più semanticamente vicini a un determinato topic
    dal Knowledge Graph. Utile per verificare la coerenza di nuove ricette
    con contenuti già pubblicati, o per creare collegamenti editoriali.
    """
    print(f"[TOOL] get_claim_pertinenti chiamato per: '{topic}'")
    try:
        claims = kg_client.get_claim_pertinenti(topic)
        if not claims:
            print("[TOOL] Nessun claim trovato.")
            return ""

        risultato = "=== CLAIM PERTINENTI ===\n"
        for i, c in enumerate(claims, 1):
            risultato += f"{i}. [{c['topic_correlato']}] (sim: {c['similarità']})\n   \"{c['claim']}\"\n\n"

        return risultato.strip()
    except Exception as e:
        print(f"[TOOL] ERRORE: {e}")
        return f"ERRORE: Impossibile recuperare i claim: {str(e)}"


@tool
def get_claim_per_retrieval(nome_elemento: str) -> str:
    """Recupera claim tecnici direttamente dai nodi Claim per arricchire la query RAG."""
    print(f"[TOOL] get_claim_per_retrieval chiamato per: '{nome_elemento}'")
    try:
        claims = kg_client.get_claim_per_retrieval(nome_elemento)
        if not claims:
            print("[TOOL] Nessun claim tecnico trovato.")
            return f"NESSUN_CLAIM_TECNICO trovato nel grafo per {nome_elemento}. Sii creativo."

        risultato = "=== CLAIM TECNICI PERTINENTI ===\n"
        for i, c in enumerate(claims, 1):
            risultato += f"{i}. [{c['topic_correlato']}] (sim: {c['similarita']})\n   \"{c['claim']}\"\n\n"
        print(f"[TOOL] Restituiti {len(claims)} claim tecnici.")
        print(f"[TOOL] Risultato:\n{risultato}")
        return risultato.strip()
    except Exception as e:
        print(f"[TOOL] ERRORE: {e}")
        return f"ERRORE: {e}"


@tool
def get_ricetta_dal_grafo(nome_elemento: str) -> str:
    """
    Verifica se una ricetta o sottoricetta è già stata pubblicata dal blog
    e ne recupera gli ingredienti (con le dosi esatte) e il procedimento.

    USA QUESTO TOOL come PRIMO CHECK prima di cercare nel DB locale
    o sul web. Se il blog ha già pubblicato questa preparazione,
    i dati qui presenti sono quelli ufficiali del blog — usali
    direttamente senza cercare altre fonti per garantire coerenza.

    Args:
        nome_elemento: Il nome della ricetta o sottoricetta da cercare.
    """
    print(f"[TOOL] get_ricetta_dal_grafo chiamato per: '{nome_elemento}'")
    risultato = kg_client.get_ricetta_completa_da_grafo(nome_elemento)

    if not risultato:
        return f"ASSENTE_DAL_GRAFO: '{nome_elemento}' non è ancora nel blog. Procedi con DB locale o web."

    # Formattazione degli ingredienti
    ingredienti_str = "\n".join(
        [
            f"  - {ing['nome']}: {ing['quantita']}"
            for ing in risultato["ingredienti"]
            if ing.get("nome") and ing.get("quantita")
        ]
    )

    # Estrazione procedimento e fonte
    procedimento_str = risultato.get(
        "procedimento", "Nessun procedimento testuale trovato."
    )
    fonti_str = ", ".join(risultato.get("fonti", ["Knowledge Graph Locale"]))

    # Testo finale da passare all'LLM
    messaggio_finale = (
        f"TROVATA_NEL_GRAFO: '{risultato['ricetta']}' è già stata pubblicata dal blog.\n"
        f"USA QUESTI DATI — sono quelli ufficiali del blog:\n"
        f"--- INGREDIENTI ---\n"
        f"{ingredienti_str}\n"
        f"--- PROCEDIMENTO ---\n"
        f"{procedimento_str}\n"
        f"--- FONTE DA CITARE ---\n"
        f"{fonti_str}\n"
        f"-------------------\n"
        f"NON cercare questa preparazione altrove e NON inventare passaggi. "
        f"RICORDA: Inserisci rigorosamente la FONTE DA CITARE nell'elenco 'fonti' finale."
    )

    # Stampiamo a schermo per il debug
    print(f"[TOOL] Risultato inviato all'agente:\n{messaggio_finale}")

    return messaggio_finale
