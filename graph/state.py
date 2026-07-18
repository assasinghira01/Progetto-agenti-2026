import operator
from typing import List, Optional, Annotated
from langgraph.graph import MessagesState
from graph.schemas import TopicPianificato, RecipeDraft


def replace_or_add(old: list, new: list) -> list:
    """Se new è una lista con un solo elemento None, svuota. Altrimenti accumula."""
    if new == [None]:
        return []
    return old + new


class Blog_Cucina(MessagesState):
    input_utente: str
    topic_originale: Optional[str] = None
    topic_corrente: Optional[str] = None
    nodo_chiamante: str
    reasoning_trace: Annotated[list[str], operator.add]
    blacklist_topics: list[str]
    richiede_variante: Optional[bool] = None
    kg_context: Annotated[List[str], operator.add]
    rag_documents: Annotated[List[str], replace_or_add]
    kg_documents: Annotated[List[str], replace_or_add]
    web_documents: Annotated[List[str], replace_or_add]
    approved_web_documents: list[str]
    approved_db_documents: list[str]
    post_draft: Optional[str] = None
    is_valid: Optional[bool] = None
    is_rigenera: Optional[bool] = False
    user_approval: Optional[str] = None
    human_feedback: Optional[str] = None
    recipe_draft: Optional[RecipeDraft] = None
    piano_editoriale: list[TopicPianificato] = []
    indice_post_corrente: int
