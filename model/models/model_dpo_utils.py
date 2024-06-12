from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
import torch

K = 1
PROMPT_TEMPLATE = """
Answer the question. You may utilize the following context:

{context}

---

{question}
"""

def apply_template(queries: list[str], db) -> list[str]:
    """
    Query the RAG database and find the most similar contexts.
    Then, create new prompts using the original queries and the retrieved context, by using the PROMPT_TEMPLATE.
    """
    llm_prompts = []
    for query in queries:
        # Search the DB.
        results = db.similarity_search_with_score(query, k=K)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = PROMPT_TEMPLATE.format(context=context_text, question=query)
        llm_prompts.append(prompt)

    return llm_prompts
