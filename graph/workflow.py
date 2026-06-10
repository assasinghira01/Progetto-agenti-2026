from langgraph.graph import StateGraph, START, END
from graph import state
from graph.schemas import PianoEditoriale
from graph.state import Blog_Cucina
from langchain_core.messages import AIMessage
from config import llm
from graph.nodes import (
    planner_node,
    krag_research_node,
    writer_node,
    validator_node,
    human_review_node,
    kg_update_node,
    aggiorna_topic_node,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from config import lista_tools  #

# --- 1. IL NODO DEGLI STRUMENTI ---
# LangGraph crea automaticamente il nodo che esegue le funzioni Python
nodo_strumenti = ToolNode(lista_tools)


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
def router_pianificazione(state: Blog_Cucina):
    """Decide se il planner ha già generato un piano editoriale o se deve ancora farlo."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print(f"  L'Agente richiede i tool: {[t['name'] for t in last_msg.tool_calls]}")
        return "tools"
    print(
        " L'Agente ha concluso la pianificazione. Passo all'estrazione del piano editoriale."
    )
    return "estrai_piano"


def estrai_piano_node(state: Blog_Cucina):
    ultimo_testo = state["messages"][-1].content
    llm_structured = llm.with_structured_output(PianoEditoriale)
    piano = llm_structured.invoke(ultimo_testo)
    print("\n--- PIANO EDITORIALE ESTRATTO ---")
    for i, post in enumerate(piano.sequenza_post):
        print(f"[{i+1}] Topic: {post.topic_ricetta} | Tipo: {post.tipo_post}")
        print(f"    Giustificazione: {post.giustificazione}")
    return {"piano_editoriale": piano.sequenza_post, "indice_post_corrente": 0}


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
    print(" L'Agente ha concluso la ricerca. Passo al validatore.")
    return "validator"


def router_dopo_tools(state: Blog_Cucina):
    """
    Decide a quale agente restituire i risultati dei tool.
    """
    if not state.get("piano_editoriale"):
        print(" [ROUTING] Risultati del tool inviati al Planner.")
        return "planner"
    print(" [ROUTING] Risultati del tool inviati allo smistamento documenti.")
    return "smista_documenti"


def after_tools(state: Blog_Cucina):
    """
    Controlla se tra i tool appena eseguiti c'era 'chiedi_variante'.
    In quel caso devia su 'aggiorna_topic', altrimenti rimanda ad analizzare i dati.
    """
    ultimo_ai_message = None
    # Andiamo a ritroso nella cronologia per trovare l'ultimo messaggio dell'AI con dei tool
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.tool_calls:
            ultimo_ai_message = m
            break

    if ultimo_ai_message:
        nomi_tool_chiamati = [t["name"] for t in ultimo_ai_message.tool_calls]
        if "chiedi_variante" in nomi_tool_chiamati:
            print(" [ROUTING] Rilevato tool 'chiedi_variante'. Vado ad aggiorna_topic.")
            return "aggiorna_topic"

    print(" [ROUTING] Tool standard eseguito. Torno all'agente di ricerca.")
    return "research"


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
builder.add_node("validator", validator_node)
builder.add_node("writer", writer_node)
builder.add_node("human_review", human_review_node)
builder.add_node("kg_update", kg_update_node)
builder.add_node("tools", nodo_strumenti)
builder.add_node("smista_documenti", smista_documenti_node)
builder.add_node("aggiorna_topic", aggiorna_topic_node)

# Definizione degli archi standard
builder.add_edge(START, "planner")
builder.add_conditional_edges(
    "planner",
    router_pianificazione,
    {"tools": "tools", "estrai_piano": "estrai_piano"},
)
builder.add_conditional_edges(
    "research",
    router_ricerca,
    {"tools": "tools", "validator": "validator"},
)

builder.add_conditional_edges(
    "tools",
    router_dopo_tools,
    {
        "planner": "planner",
        "smista_documenti": "smista_documenti",
    },
)

builder.add_conditional_edges(
    "smista_documenti",
    after_tools,
    {"aggiorna_topic": "aggiorna_topic", "research": "research"},
)


builder.add_edge("aggiorna_topic", "research")


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
