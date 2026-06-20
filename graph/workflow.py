import json
from typing import Literal

from langgraph.graph import StateGraph, START, END
from graph.schemas import PianoEditoriale, TopicPianificato
from graph.state import Blog_Cucina
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from config import llm
from graph.nodes import (
    human_review_variante,
    planner_node,
    krag_research_node,
    writer_node,
    validator_node,
    human_review_node,
    kg_update_node,
    human_review_planner,
)
from langgraph.checkpoint.memory import InMemorySaver
from config import lista_tools  #

# --- 1. IL NODO DEGLI STRUMENTI ---


tools_by_name = {tool.name: tool for tool in lista_tools}


# --- 2. FUNZIONI DI ROUTING ---
def smista_documenti_node(state: Blog_Cucina):
    print("\n--- [NODO INTERMEDIO: SMISTAMENTO DOCUMENTI] ---")
    messaggi = state.get("messages", [])

    rag_docs = []
    web_docs = []

    for msg in messaggi:
        if getattr(msg, "type", "") == "tool":
            if msg.name == "cerca_ricetta_nel_db":
                # Aggiungiamo solo se non è un messaggio di errore
                if "Errore" not in msg.content and "Nessuna ricetta" not in msg.content:

                    try:
                        lista_ricette = json.loads(msg.content)
                        rag_docs.extend(lista_ricette)
                    except json.JSONDecodeError:
                        pass

            elif msg.name == "esegui_ricerca_web":
                if "Nessuna ricetta" not in msg.content:
                    lista_ricette_web = msg.content.split("\n\n|||SPLIT_DOC|||\n\n")
                    web_docs.extend(lista_ricette_web)

    print(
        f" -> [STATO] Estratti {len(rag_docs)} documenti RAG e {len(web_docs)} documenti WEB."
    )
    messaggi_da_cancellare = [RemoveMessage(id=m.id) for m in state["messages"]]
    return {
        "rag_documents": rag_docs,
        "web_documents": web_docs,
        "messages": messaggi_da_cancellare,
    }


# --- 3. FUNZIONI DI ROUTING CONDIZIONALE (ROUTER) ---
def router_pianificazione(state: Blog_Cucina) -> Literal["tools", "estrai_piano"]:
    last_message = state["messages"][-1]
    input_utente = state["input_utente"]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    if input_utente == "PIANIFICAZIONE_AUTOMATICA":
        return "estrai_piano"
    else:
        return "estrai_singolo_topic"


def estrai_piano_node(state: Blog_Cucina):
    print("\n--- [NODO: ESTRAZIONE PIANO FINALE] ---")
    storico_riflessioni = state.get("reasoning_trace", [])
    feedback = state.get("human_feedback", "")

    parti = feedback.split("|")
    piano_salvato = parti[1].replace("piano:", "").strip() if len(parti) > 1 else ""
    testo_ragionamenti = "\n".join(storico_riflessioni)

    contesto_modifica = ""
    if feedback and "modifica:" in feedback:
        contesto_modifica = f"""
        ATTENZIONE: Questo piano è il risultato di una MODIFICA. 
        L'agente ha deliberatamente mantenuto intatti un topic del piano precedente e ne ha appena trovato uno nuovo.
    piano_salvato = (
        Devi scansionare il testo per trovare la lista unificata (i topic mantenuti + il nuovo topic). i topic da considerare sono: {piano_salvato}.
        """

    prompt_estrazione = f"""
    Leggi attentamente il seguente log dei ragionamenti di un agente AI.
    Il tuo compito è estrarre il PIANO EDITORIALE FINALE completo e aggiornato.
    {contesto_modifica}
    
    🚨 REGOLA MATEMATICA ASSOLUTA: 
    Il piano estratto DEVE contenere SEMPRE E RIGOROSAMENTE 3 RICETTE. Non 1, non 2, ma ESATTAMENTE 3.
    Cerca l'ultima riflessione dell'agente in cui elenca il totale definitivo dei piatti approvati.
    
    Estrai per ognuno dei 3 topic finali:
    - Il nome della ricetta
    - La categoria (Antipasto, Primo, Secondo, o Dolce)
    - Una  giustificazione sul perché fa parte del piano.
    
    LOG RAGIONAMENTI:
    {testo_ragionamenti}
    """

    llm_structured = llm.with_structured_output(PianoEditoriale)
    piano = llm_structured.invoke(prompt_estrazione)

    return {
        "piano_editoriale": piano,
        "indice_post_corrente": 0,
        "reasoning_trace": storico_riflessioni,
    }


def estrai_singolo_topic_node(state: Blog_Cucina):
    print("\n--- [NODO: ESTRAZIONE TOPIC / GESTIONE FEEDBACK] ---")
    storico_riflessioni = state.get("reasoning_trace", [])
    feedback = state.get("human_feedback", "")
    # =================================================================
    # CASO 1: Override Umano
    # =================================================================
    if feedback:
        print(f" -> Rilevato feedback umano: '{feedback}'")
        prompt_feedback = f"""
        L'utente ha richiesto una modifica manuale: "{feedback}".
        Estrai il nome del nuovo piatto, assegnali la categoria corretta e lascia la giustificazione vuota.
        """

        llm_estrazione = llm.with_structured_output(TopicPianificato)
        risultato = llm_estrazione.invoke(prompt_feedback)

        print(
            f" -> [Override Umano] Topic: {risultato.topic} | Categoria: {risultato.categoria}"
        )

        return {
            "topic_corrente": risultato.topic,
            "human_feedback": "",
        }

    testo_ragionamenti = "\n".join(storico_riflessioni)
    is_variante = (
        "variante" in testo_ragionamenti.lower()
        or "bloccato" in testo_ragionamenti.lower()
    )

    if is_variante:
        print(" -> Rilevato duplicato storico. Estrazione della VARIANTE...")
        prompt_estrazione = f"""
        L'agente ha proposto una VARIANTE a causa di un duplicato.
        Leggi i log ed estrai:
        1. Il topic (la variante approvata).
        2. La categoria gastronomica.
        3. La giustificazione (spiega brevemente perché ha scelto questa variante).
        LOG: {testo_ragionamenti}
        
        """

        llm_estrazione_singola = llm.with_structured_output(TopicPianificato)
        risultato = llm_estrazione_singola.invoke(prompt_estrazione)

        print(
            f" -> [Estrazione] Topic: {risultato.topic} | Categoria: {risultato.categoria} | motivazione : {risultato.giustificazione}"
        )

        return {
            "topic_corrente": risultato.topic,
            "reasoning_trace": storico_riflessioni,
            "richiede_variante": True,
        }

    else:
        print(" -> Topic originale approvato. Estrazione in corso...")
        prompt_estrazione = f"""
        L'agente ha approvato il topic originale.
        Leggi i log ed estrai:
        1. Il topic confermato.
        2. La categoria gastronomica.
        3. Lascia la giustificazione vuota.
        LOG: {testo_ragionamenti}
        """

        llm_estrazione_singola = llm.with_structured_output(TopicPianificato)
        risultato = llm_estrazione_singola.invoke(prompt_estrazione)
        messaggi_da_cancellare = [RemoveMessage(id=m.id) for m in state["messages"]]

        print(
            f" -> [Estrazione] Topic: {risultato.topic} | Categoria: {risultato.categoria}"
        )

        return {
            "messages": messaggi_da_cancellare,
            "topic_corrente": risultato.topic,
            "reasoning_trace": storico_riflessioni,
        }


def router_dopo_estrazione(
    state: Blog_Cucina,
) -> Literal["human_review_variante", "research"]:
    print("\n--- [ROUTER: CONTROLLO DUPLICATI / HITL] ---")

    if state.get("richiede_variante", False) == True:
        print(
            " -> Rilevato DUPLICATO (Variante proposta). Invio a HUMAN_REVIEW_VARIANTE."
        )
        return "human_review_variante"
    else:
        print(" -> Topic ORIGINALE e approvato. Vado diretto in RESEARCH.")
        return "research"


def router_ricerca(state: Blog_Cucina):
    """Decide se l'agente deve usare un tool (ReAct) o se ha finito la ricerca."""
    ultimo_messaggio = state["messages"][-1]

    # Se l'LLM ha richiesto l'uso di uno strumento (es. ha generato tool_calls)
    if hasattr(ultimo_messaggio, "tool_calls") and ultimo_messaggio.tool_calls:
        print(
            f"  L'Agente richiede i tool: {[t['name'] for t in ultimo_messaggio.tool_calls]}"
        )
        return "tools"
    # Se non ci sono tool_calls, l'agente ha finito di raccogliere dati

    return "validator"


def tool_node(state: Blog_Cucina):
    last_message = state["messages"][-1]
    nodo_chiamante = state["nodo_chiamante"]
    tool_calls = last_message.tool_calls
    print(f"\n--- [TOOL NODE] Trovate {len(tool_calls)} chiamate a tool ---")
    observations = []
    ragionamenti_aggiunti = []

    for idx, tool_call in enumerate(tool_calls, start=1):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name != "think_tool":
            print(f"\n[{idx}] Esecuzione tool: {tool_name}")
            print(f"    Argomenti: {tool_args}")

        tool = tools_by_name.get(tool_name)
        if tool is None:
            msg = f"Errore: tool '{tool_name}' non trovato"
            observations.append(msg)
        else:
            result = tool.invoke(tool_args)
            observations.append(result)

            if tool_name == "think_tool":
                # Puliamo il prefisso tecnico
                testo_pulito = result.replace(
                    "Riflessione registrata con successo: ", ""
                ).strip()
                print("\n" + "=" * 60)
                print(" 🧠 RAGIONAMENTO AGENTE IN CORSO...")
                print("-" * 60)
                print(f" {testo_pulito}")
                print("=" * 60 + "\n")
                ragionamento = f"[{nodo_chiamante.upper()}] {testo_pulito}"
                # Aggiungi alla lista dei ragionamenti
                ragionamenti_aggiunti.append(ragionamento)

    tool_outputs = [
        ToolMessage(
            content=str(obs), name=tool_call["name"], tool_call_id=tool_call["id"]
        )
        for obs, tool_call in zip(observations, tool_calls)
    ]

    return {"messages": tool_outputs, "reasoning_trace": ragionamenti_aggiunti}


def router_dopo_tools(state: Blog_Cucina) -> str:
    # 1. Recuperiamo chi ha chiamato il tool e i messaggi
    print("\n--- [ROUTER DOPO TOOLS] ---")
    nodo_chiamante = state.get("nodo_chiamante")
    messaggi = state.get("messages", [])
    input_utente = state.get("input_utente", "")
    if not messaggi:
        return nodo_chiamante  # Fallback di sicurezza

    ultimo_messaggio = messaggi[-1]

    # 2. LOGICA INTELLIGENTE: L'ultimo tool usato è stato il "think_tool"?
    if (
        isinstance(ultimo_messaggio, ToolMessage)
        and ultimo_messaggio.name == "think_tool"
    ):
        # Mettiamo tutto in maiuscolo per evitare errori se l'LLM scrive "Stato: Finito"
        riflessione = ultimo_messaggio.content.upper()

        # A. L'agente usa la parola d'ordine di fine lavoro
        if "STATO: FINITO" in riflessione:

            if nodo_chiamante == "research":
                return "smista_documenti"

            elif nodo_chiamante == "planner":
                if input_utente == "PIANIFICAZIONE_AUTOMATICA":
                    return "estrai_piano"
                else:

                    return "estrai_singolo_topic"

        else:
            return nodo_chiamante

    return nodo_chiamante


def check_validation(state: Blog_Cucina):
    """Legge l'esito del validatore strutturato e fa routing."""
    messaggi = state.get("messages", [])

    if messaggi:
        ultimo_messaggio = messaggi[-1]

        if hasattr(ultimo_messaggio, "tool_calls") and ultimo_messaggio.tool_calls:
            print(" ROUTING: Il validatore sta riflettendo... Vado ai tools.")
            return "tools"

    if state.get("is_valid", False) == True:
        print("ROUTING: Validazione superata! Procedo alla scrittura della bozza.")
        return "writer"
    else:
        print(
            "ROUTING: Rilevata assurdità o mancanza totale di dati. Blocco il processo!"
        )
        return "end_block"


def altri_post_da_fare(state):

    indice = state["indice_post_corrente"]

    totale = len(state["piano_editoriale"])

    if indice < totale:
        return "research"

    return END


builder = StateGraph(Blog_Cucina)

# Registrazione dei nodi
builder.add_node("planner", planner_node)
builder.add_node("estrai_piano", estrai_piano_node)
builder.add_node("research", krag_research_node)
builder.add_node("human_review_planner", human_review_planner)
builder.add_node("human_review_variante", human_review_variante)
builder.add_node("validator", validator_node)
builder.add_node("writer", writer_node)
builder.add_node("human_review", human_review_node)
builder.add_node("estrai_singolo_topic", estrai_singolo_topic_node)
builder.add_node("kg_update", kg_update_node)
builder.add_node("tools", tool_node)
builder.add_node("smista_documenti", smista_documenti_node)


# Definizione degli archi standard
builder.add_edge(START, "planner")

builder.add_conditional_edges(
    "planner",
    router_pianificazione,
    {
        "tools": "tools",
        "estrai_piano": "estrai_piano",
        "estrai_singolo_topic": "estrai_singolo_topic",
    },
)

builder.add_conditional_edges(
    "tools",
    router_dopo_tools,
    {
        "planner": "planner",
        "research": "research",
        "estrai_piano": "estrai_piano",
        "smista_documenti": "smista_documenti",
        "estrai_singolo_topic": "estrai_singolo_topic",
        "validator": "validator",
    },
)

builder.add_conditional_edges(
    "estrai_singolo_topic",
    router_dopo_estrazione,
    {"human_review_variante": "human_review_variante", "research": "research"},
)


builder.add_edge("estrai_piano", "human_review_planner")


builder.add_conditional_edges(
    "research",
    router_ricerca,
    {"tools": "tools", "validator": "validator"},
)


builder.add_edge("smista_documenti", "validator")
# Definizione dell'arco condizionale dal validatore
builder.add_conditional_edges(
    "validator",
    check_validation,
    {
        "tools": "tools",
        "writer": "writer",
        "end_block": END,  # Se è falso, interrompe l'esecuzione evitando allucinazioni
    },
)


# Dal Writer andiamo alla Revisione Umana
builder.add_edge("writer", "human_review")

# L'arco condizionale del feedback umano (Loop di Correzione)


# Dal salvataggio andiamo alla fine
builder.add_edge("kg_update", END)
memoria_temporanea = InMemorySaver()
app = builder.compile(checkpointer=memoria_temporanea)

# with open("diagramma_agente.png", "wb") as f:
# f.write(app.get_graph().draw_mermaid_png())
