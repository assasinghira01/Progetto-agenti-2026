from typing import Literal

from langgraph.graph import StateGraph, START, END
from graph import state
from graph.schemas import PianoEditoriale, TopicEstratto
from graph.state import Blog_Cucina
from langchain_core.messages import AIMessage, ToolMessage
from config import llm
from graph.nodes import (
    human_review_variante,
    planner_node,
    krag_research_node,
    writer_node,
    validator_node,
    human_review_node,
    kg_update_node,
    aggiorna_topic_node,
    human_review_planner,
)
from langgraph.checkpoint.memory import InMemorySaver
from config import lista_tools  #

# --- 1. IL NODO DEGLI STRUMENTI ---


tools_by_name = {tool.name: tool for tool in lista_tools}


# --- 2. FUNZIONI DI ROUTING ---
def smista_documenti_node(state: Blog_Cucina):
    print("\n--- [NODO INTERMEDIO: SMISTAMENTO DOCUMENTI] ---")
    ultimo_messaggio = state["messages"][-1]
    # Controlliamo quale tool ha appena risposto per salvare il testo nel cassetto corretto
    if hasattr(ultimo_messaggio, "name"):
        if ultimo_messaggio.name == "cerca_ricetta_nel_db":
            print(" -> [STATO] Salvo i dati in: rag_documents")
            return {"rag_documents": [ultimo_messaggio.content]}
        elif ultimo_messaggio.name == "esegui_ricerca_web":
            print(" -> [STATO] Salvo i dati in: web_documents")
            return {"web_documents": [ultimo_messaggio.content]}

    return {}


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
    messaggi = state["messages"]

    storico_riflessioni = []

    for msg in messaggi:
        # Cerchiamo solo i messaggi di risposta generati dal think_tool
        if getattr(msg, "name", "") == "think_tool":
            # Rimuoviamo il prefisso tecnico per avere solo il testo pulito
            testo_pulito = msg.content.replace(
                "Riflessione registrata con successo: ", ""
            ).strip()
            storico_riflessioni.append(testo_pulito)

    testo_ragionamenti = "\n".join(storico_riflessioni)

    prompt_estrazione = f"""
    Leggi attentamente il seguente log dei ragionamenti di un agente AI.
    Estrai ESATTAMENTE i 3 topic finali che l'agente ha deciso di approvare.
    
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
    messaggi = state.get("messages", [])

    print("\n--- [NODO: ESTRAZIONE TOPIC FINALE VARIANTE] ---")

    storico_riflessioni = []

    # Raccogliamo in silenzio tutti i log del think_tool
    for msg in messaggi:
        if getattr(msg, "name", "") == "think_tool":
            testo_pulito = msg.content.replace(
                "Riflessione registrata con successo: ", ""
            ).strip()
            storico_riflessioni.append(testo_pulito)

    testo_ragionamenti = "\n".join(storico_riflessioni)

    prompt_estrazione = f"""
    Leggi attentamente il seguente log dei ragionamenti di un agente AI.
    L'agente stava cercando un singolo topic. Potrebbe aver scartato l'idea iniziale e proposto una variante.
    Estrai ESATTAMENTE il nome dell'UNICA ricetta finale (o variante) che l'agente ha deciso di approvare alla fine (quando dichiara STATO: FINITO).
    
    LOG RAGIONAMENTI:
    {testo_ragionamenti}
    """

    # Invochiamo l'LLM per estrarre il dato pulito
    llm_estrazione_singola = llm.with_structured_output(TopicEstratto)
    risultato = llm_estrazione_singola.invoke(prompt_estrazione)

    topic_finale = risultato.topic.strip().capitalize()

    print(f" -> [Estrazione] Variante estratta e salvata nello stato: {topic_finale}")

    # Sovrascriviamo il vecchio topic nello stato di LangGraph restituendo una stringa pura
    return {
        "topic_corrente": topic_finale,
        "reasoning_trace": storico_riflessioni,
    }


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
    tool_calls = last_message.tool_calls
    print(f"\n--- [TOOL NODE] Trovate {len(tool_calls)} chiamate a tool ---")
    observations = []
    for idx, tool_call in enumerate(tool_calls, start=1):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # STAMPA BASE per i tool normali (opzionale, la teniamo per debug)
        if tool_name != "think_tool":
            print(f"\n[{idx}] Esecuzione tool: {tool_name}")
            print(f"    Argomenti: {tool_args}")

        tool = tools_by_name.get(tool_name)
        if tool is None:
            msg = f"Errore: tool '{tool_name}' non trovato"
            observations.append(msg)
        else:
            # Esegui il tool
            result = tool.invoke(tool_args)
            observations.append(result)

            # --- LA NOSTRA STAMPA UNIVERSALE DEI RAGIONAMENTI ---
            if tool_name == "think_tool":
                # Puliamo il prefisso tecnico per l'utente finale
                testo_pulito = result.replace(
                    "Riflessione registrata con successo: ", ""
                ).strip()
                print("\n" + "=" * 60)
                print(" 🧠 RAGIONAMENTO AGENTE IN CORSO...")
                print("-" * 60)
                print(f" {testo_pulito}")
                print("=" * 60 + "\n")

    # Costruzione dei messaggi di risposta
    tool_outputs = [
        ToolMessage(
            content=str(obs), name=tool_call["name"], tool_call_id=tool_call["id"]
        )
        for obs, tool_call in zip(observations, tool_calls)
    ]

    return {"messages": tool_outputs}


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
        # A. L'agente usa la parola d'ordine di fine lavoro
        if "STATO: FINITO" in riflessione:

            if nodo_chiamante == "research":
                return "smista_documenti"
            elif nodo_chiamante == "planner":
                if input_utente == "PIANIFICAZIONE_AUTOMATICA":
                    return "estrai_piano"
                else:
                    return "estrai_singolo_topic"
        # B. L'agente non ha finito (ha usato "STATO: CONTINUO" o deve ancora validare)
        else:
            return nodo_chiamante
    return nodo_chiamante


def check_validation(state: Blog_Cucina):
    """Legge l'esito del validatore strutturato e fa routing."""
    if state.get("is_valid") == True:
        print("ROUTING: Validazione superata! Procedo alla scrittura della bozza.")
        return "writer"
    else:
        print(
            "ROUTING: Rilevata assurdità o mancanza totale di dati. Blocco il processo!"
        )
        return "end_block"


def check_human_approval(state: Blog_Cucina):
    """Legge la decisione umana presa durante l'interruzione."""
    feedback = state.get("human_feedback", "").lower()

    if "approvo" in feedback or "ok" in feedback or "perfetto" in feedback:
        print(" Semaforo VERDE umano: Procedo al salvataggio su Neo4j.")
        return "kg_update"
    else:
        print(" Semaforo ARANCIONE umano: L'utente vuole modifiche. Riscrivo il post!")
        return "writer"


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
builder.add_node("aggiorna_topic", aggiorna_topic_node)


# Definizione degli archi standard
builder.add_edge(START, "planner")

builder.add_conditional_edges(
    "planner",
    router_pianificazione,
    {
        "tools": "tools",
        "estrai_piano": "estrai_piano",
        "human_review_variante": "human_review_variante",
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
        "human_review_variante": "human_review_variante",
        "estrai_singolo_topic": "estrai_singolo_topic",
    },
)

builder.add_edge("estrai_singolo_topic", "human_review_variante")
builder.add_edge("estrai_piano", "human_review_planner")

builder.add_conditional_edges(
    "research",
    router_ricerca,
    {"tools": "tools", "validator": "validator"},
)


builder.add_edge("smista_documenti", "research")

# Definizione dell'arco condizionale dal validatore
builder.add_conditional_edges(
    "validator",
    check_validation,
    {
        "writer": "writer",
        "end_block": END,  # Se è falso, interrompe l'esecuzione evitando allucinazioni
    },
)

# Dal Writer andiamo alla Revisione Umana
builder.add_edge("writer", "human_review")

# L'arco condizionale del feedback umano (Loop di Correzione)
builder.add_conditional_edges(
    "human_review", check_human_approval, {"kg_update": "kg_update", "writer": "writer"}
)

# Dal salvataggio andiamo alla fine
builder.add_edge("kg_update", END)
memoria_temporanea = InMemorySaver()
app = builder.compile(checkpointer=memoria_temporanea)

with open("diagramma_agente.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())
