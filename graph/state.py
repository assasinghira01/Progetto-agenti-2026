from typing import List, Optional
from langgraph.graph import MessagesState

class Blog_Cucina(MessagesState):
    input_utente: str
    topic_corrente: Optional[str]
    kg_context: List[str]                
    rag_documents: List[str]              
    post_draft: Optional[str]
    web_documents: List[str]              
    post_draft: Optional[str]
    is_valid: bool           
    user_approval: str