from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
import torch
import re

K = 1
PROMPT_TEMPLATE = """
Answer the question. You may utilize the following context:

{context}

---

{question}

---

Remember that this is a multiple-choice question. The correct answer is a single letter (A, B, C, D, or E). Answer in the format "The correct answer is : (letter)"
"""

def extract_options(question):
    # Regex pattern to find options labeled A, B, C, D, etc., followed by a description
    pattern = r'([A-Z])\.\s+(.+?)(?=\n[A-Z]\.|$)'
    matches = re.findall(pattern, question, re.DOTALL)
    options = {match[0]: match[1].strip() for match in matches}
    return options

def extract_answer(text, options):
    pattern = r'(?i)\b(?:the\s+)?(?:correct|right)\s+answer\s+is\s*:?\s*([A-Z])'
    match = re.search(pattern, text)
    if match:
        answer_letter = match.group(1).upper()
        if answer_letter in options:
            return answer_letter
    # If not directly mentioned, look if the description matches any of the options
    explanation_index = text.lower().find("explanation:")
    if explanation_index != -1:
        # Only consider text after "Explanation:"
        text = text[explanation_index:]
    for letter, description in options.items():
        if description.lower() in text.lower():
            return letter.upper()
    return "No answer found"

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
