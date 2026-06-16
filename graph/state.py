import operator
from typing import List, Optional, Annotated
from langgraph.graph import MessagesState
from graph.schemas import TopicPianificato, RecipeDraft


class Blog_Cucina(MessagesState):
    input_utente: str
    topic_corrente: Optional[str] = None
    nodo_chiamante: str
    reasoning_trace: Annotated[list[str], operator.add]
    blacklist_topics: list[str]
    kg_context: Annotated[List[str], operator.add]
    rag_documents: Annotated[List[str], operator.add]
    web_documents: Annotated[List[str], operator.add]
    approved_web_documents: list[str]
    approved_db_documents: list[str]
    post_draft: Optional[str] = None
    is_valid: Optional[bool] = None
    user_approval: Optional[str] = None
    human_feedback: Optional[str] = None
    recipe_draft: Optional[RecipeDraft] = None
    piano_editoriale: list[TopicPianificato] = []
    indice_post_corrente: int
