#  Valutazione automatica con OpenAI
import os
import sys
import uuid
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from langsmith import Client
from langchain_openai import ChatOpenAI
from graph.workflow import app
from langsmith.evaluation import evaluate
from pydantic import BaseModel, Field
from langgraph.types import Command

client = Client()


# Configurazione del valutatore (OpenAI)
def _build_evaluator_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


evaluator_llm = _build_evaluator_llm()


def _graded_invoke(schema, prompt_text):
    try:
        return evaluator_llm.with_structured_output(schema).invoke(prompt_text)
    except Exception as e:
        print(f"⚠️  Errore giudice OpenAI: {e}")
        return None


# Funzione target per LangSmith
def run_blogger_until_draft(inputs: Dict[str, Any]) -> Dict[str, Any]:
    user_input = inputs.get("user_input", "")
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    snapshot = None
    steps_used = 0

    try:
        app.invoke({"input_utente": user_input}, config)
    except Exception as e:
        print(f"\n❌ ERRORE FATALE AGENTE in input '{user_input}': {e}")
        return {"draft_content": "", "error": str(e)}

    # Loop per superare i gate e arrivare alla bozza
    MAX_STEPS = 50
    for _ in range(MAX_STEPS):
        steps_used += 1
        snapshot = app.get_state(config)
        next_nodes = snapshot.next if snapshot else ()

        if not next_nodes:
            break
        if "human_review_node" in next_nodes:
            break
        if "human_review_planner" in next_nodes:
            try:
                app.invoke(Command(resume="APPROVA"), config)
                continue
            except Exception:
                break
        if "human_review_variante" in next_nodes:
            try:
                app.invoke(Command(resume="APPROVA"), config)
                continue
            except Exception:
                break
        else:
            print(
                f"⚠️ Stato inatteso: il grafo è fermo su {next_nodes} senza essere un nodo HITL noto."
            )
            break
    if steps_used >= MAX_STEPS and (not snapshot or snapshot.next):
        print(f"⚠️ Raggiunto MAX_STEPS senza completare per input: {user_input}")
    try:
        state = app.get_state(config)
        values = state.values if state else {}
    except Exception:
        values = {}

    draft = values.get("post_draft", "") or ""
    if not draft:
        messages = values.get("messages", [])
        if messages:
            draft = getattr(messages[-1], "content", "") or ""

    db_sources = values.get("approved_db_documents", []) or []
    web_sources = values.get("approved_web_documents", []) or []
    all_sources = db_sources + web_sources

    tools_called = []
    for m in values.get("messages", []):
        for tc in getattr(m, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name and name not in tools_called:
                tools_called.append(name)
    if values.get("rag_documents"):
        tools_called.append("cerca_ricetta_nel_db")
    if values.get("web_documents"):
        tools_called.append("esegui_ricerca_web")

    return {
        "draft_content": draft,
        "sources": all_sources,
        "local_sources": db_sources,
        "tools_called": tools_called,
        "user_input": user_input,
        "interrupted": bool(snapshot and snapshot.next),
        "error": None,
    }


# Evaluators


class QualityScore(BaseModel):
    score: int = Field(description="Punteggio da 1 a 5.")
    reasoning: str = Field(description="Motivazione.")


def evaluate_quality(run, example) -> dict:
    generated_post = run.outputs.get("draft_content", "") if run.outputs else ""
    if not generated_post:
        return {"key": "Qualitative_Score", "score": 0.0, "comment": "Nessuna bozza."}
    prompt = (
        "Sei un critico gastronomico. Valuta la qualità della seguente ricetta da 1 a 5.\n"
        "Criteri: struttura chiara, tono coinvolgente, precisione dosi, assenza ripetizioni.\n\n"
        f"Articolo:\n{generated_post}\n"
    )
    result = _graded_invoke(QualityScore, prompt)
    if result is None:
        return {"key": "Qualitative_Score", "score": 0.0, "comment": "Giudice fallito."}
    return {
        "key": "Qualitative_Score",
        "score": result.score / 5.0,
        "comment": result.reasoning,
    }


class GroundingScore(BaseModel):
    factualita: int = Field(description="Da 0 a 5: quanto è fattuale.")
    reasoning: str = Field(description="Spiegazione.")


def evaluate_grounding(run, example) -> dict:
    generated_post = run.outputs.get("draft_content", "") if run.outputs else ""
    if not generated_post:
        return {"key": "Source_Grounding", "score": 0.0, "comment": "Nessuna bozza."}
    low = generated_post.lower()
    citation_markers = [
        "http",
        "**fonti utilizzate**",
        "fonti utilizzate:",
        "[fonte",
        "fonti:",
        "secondo",
        "riferimenti",
    ]
    has_citations = any(m in low for m in citation_markers)
    cit_score = 1.0 if has_citations else 0.0
    prompt = (
        "Valuta da 0 a 5 quanto questa ricetta è FATTUALE (priva di invenzioni).\n"
        f"Articolo:\n{generated_post[:6000]}\n"
    )
    result = _graded_invoke(GroundingScore, prompt)
    fact_score = 0.5
    reasoning = "(giudizio non disponibile)"
    if result is not None:
        fact_score = max(0, min(5, result.factualita)) / 5.0
        reasoning = result.reasoning
    final = round(0.5 * cit_score + 0.5 * fact_score, 2)
    return {
        "key": "Source_Grounding",
        "score": final,
        "comment": f"Citazioni: {'sì' if has_citations else 'NO'}. {reasoning}",
    }


def evaluate_failure_cases(run, example) -> dict:
    out = run.outputs or {}
    post = out.get("draft_content", "") or ""
    sources = out.get("sources") or []
    interrupted = out.get("interrupted", False)
    failures = []
    if not post.strip():
        failures.append("F1_EMPTY_DRAFT")
    if post.strip():
        low = post.lower()
        if not any(m in low for m in ["http", "[fonte", "fonti:", "riferimenti"]):
            failures.append("F2_NO_CITATIONS")
    if post.strip() and not sources:
        failures.append("F3_NO_SOURCES")
    if not post.strip() and not interrupted:
        failures.append("F4_NO_INTERRUPT")
    passed = 4 - len(failures)
    return {
        "key": "Failure_Cases",
        "score": passed / 4.0,
        "comment": " | ".join(failures) if failures else "Nessun fallimento.",
    }


def evaluate_tool_usage(run, example) -> dict:
    out = run.outputs or {}
    tools = set(out.get("tools_called") or [])
    has_search = bool(
        {"esegui_ricerca_web", "cerca_ricetta_nel_db"}.intersection(tools)
    )
    has_kg = bool({"get_ingredienti", "controlla_storico_post"}.intersection(tools))
    score = 1.0
    notes = []
    if not tools:
        score = 0.3
        notes.append("Nessun tool usato.")
    else:
        notes.append(f"Tool: {', '.join(tools)}.")
        if not has_search:
            score -= 0.2
            notes.append("Manca ricerca.")
        if not has_kg:
            score -= 0.2
            notes.append("Manca KG.")
    score = max(0.0, min(1.0, score))
    return {"key": "Tool_Usage", "score": score, "comment": " ".join(notes)}


# 4. Report e Main


def observability_report(project_name: str = None, limit: int = 50) -> dict:
    project = project_name or os.environ.get(
        "LANGSMITH_PROJECT", "BloggerCucina_CCAI2026"
    )
    try:
        all_runs = list(
            client.list_runs(project_name=project, limit=min(limit * 20, 100))
        )
    except Exception as e:
        return {"error": str(e)}
    if not all_runs:
        return {"info": f"Nessuna run in '{project}'."}
    roots = [r for r in all_runs if getattr(r, "is_root", False)] or [
        r for r in all_runs if not getattr(r, "parent_run_id", None)
    ]
    lat, err = [], 0
    for r in roots:
        if getattr(r, "start_time", None) and getattr(r, "end_time", None):
            lat.append((r.end_time - r.start_time).total_seconds())
        if getattr(r, "error", None):
            err += 1
    # breakdown per nodo/tool + token usage
    per_node_latency = {}
    total_tokens = 0
    for r in all_runs:
        nome = getattr(r, "name", "sconosciuto")
        if getattr(r, "start_time", None) and getattr(r, "end_time", None):
            dur = (r.end_time - r.start_time).total_seconds()
            per_node_latency.setdefault(nome, []).append(dur)
        tok = getattr(r, "total_tokens", None)
        if tok:
            total_tokens += tok

    breakdown = {
        nome: round(sum(durate) / len(durate), 2)
        for nome, durate in per_node_latency.items()
    }

    return {
        "project": project,
        "esecuzioni": len(roots),
        "run_totali": len(all_runs),
        "latenza_media_s": round(sum(lat) / len(lat), 2) if lat else 0,
        "tasso_errore": round(err / len(roots), 3) if roots else 0,
        "token_totali_stimati": total_tokens,
        "latenza_media_per_nodo_s": breakdown,
    }


def create_example_dataset():
    dataset_name = "BloggerCucina_Dataset_Test"
    examples = [
        {"user_input": "Scrivi un post sulla Carbonara"},
        {"user_input": "Vorrei una ricetta per il Tiramisù alle fragole"},
        {"user_input": "Voglio una ricetta sulla margherita"},
        {"user_input": "Voglio un post sull'insalata russa"},
    ]
    try:
        ds = client.read_dataset(dataset_name=dataset_name)
        print(f"  Dataset '{dataset_name}' già esistente.")
        return ds
    except Exception:
        ds = client.create_dataset(
            dataset_name=dataset_name, description="Test per agente cucina"
        )
        for ex in examples:
            client.create_example(inputs=ex, dataset_id=ds.id)
        print(f" Dataset '{dataset_name}' creato.")
        return ds


if __name__ == "__main__":
    print("=== Valutazione Agente Cucina (OpenAI Judge) ===")
    dataset = create_example_dataset()
    try:
        evaluate(
            run_blogger_until_draft,
            data=dataset.name,
            evaluators=[
                evaluate_quality,
                evaluate_grounding,
                evaluate_failure_cases,
                evaluate_tool_usage,
            ],
            experiment_prefix="BloggerCucina_Eval_OpenAI",
            max_concurrency=1,
        )
        print("✅ Valutazione completata. Vedi dashboard LangSmith.")
    except Exception as e:
        print(f"❌ Errore: {e}")
    print("\n📊 Report osservabilità:")
    for k, v in observability_report().items():
        print(f"  {k}: {v}")
