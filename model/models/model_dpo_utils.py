from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
import torch
import re

NUM_TO_LETTER = {
    '1': 'A',
    '2': 'B',
    '3': 'C',
    '4': 'D'
}

K = 1
PROMPT_TEMPLATE = """
Answer the question. You may utilize the following context:

{context}---

{question}"""
def remove_period(s: str) -> str:
    if s.endswith('.'):
        return s[:-1]  # Return the string without the last character (the period)
    return s


def extract_options(question):
    # Regex pattern to find options labeled A, B, C, D, etc., followed by a description
    pattern = r'^([A-Z])\.\s+(.+?)\n'
    matches = re.findall(pattern, question, re.MULTILINE)
    options = {match[0]: remove_period(match[1].strip()) for match in matches}
    return options


def extract_answer(text, options):
    # First we search for the correct letter/number.
    pattern = r'(?i)(?s:.*)\b(?:the\s+)?(?:correct|right|final)\s+(?:answer|letter)\s*(?:is)?\s*(?:option)?\s*:?\s*-?\s*(?:option)?\s*([A-D]|[1-4])(?![a-z])'
    match = re.search(pattern, text)
    if match:
        answer = match.group(1).upper()
        if answer in options:
            return answer
        
        if answer in NUM_TO_LETTER:
            return NUM_TO_LETTER[answer]
    
    # If not directly mentioned, look if the description matches any of the options
    result = "A"
    explanation_index = text.find("\n\nExplanation:")
    text = text[explanation_index:]

    # We try to find the answer in the explanation part
    for letter, description in options.items():
        if description.lower() in text.lower():
            result = letter.upper()

    # We try to find the answer in the answer part
    pattern = r'(?i)(the\s+)?(correct|right|final)?\s*(answer)\s*(is)?\s*:?'
    match = re.search(pattern, text)
    if match:
        text = text[match.end():]
        
        for letter, description in options.items():
            if description.lower() in text.lower():
                result = letter.upper()
        
    return result

def apply_template(query: str, db) -> str:
    """
    Query the RAG database and find the most similar contexts.
    Then, create new prompts using the original queries and the retrieved context, by using the PROMPT_TEMPLATE.
    """
    # Search the DB.
    results = db.similarity_search_with_score(query, k=K)

    max_length = 7200
    available_space = max_length - len(query)

    # Remove contexts with very low similarity
    results = [(doc, _score) for doc, _score in results if _score < 0.55]

    if len(results) == 0:
        return query
    
    context_text = ""
    total_context_length = 0
    i = 0
    # Ensure at least one context is added
    context_text += f"--- Context\n\n{results[i][0].page_content}\n\n"
    total_context_length += len(results[i][0].page_content) + len(f"--- Context\n\n")
    i += 1

    # Add additional contexts as long as they fit within the available space
    while i < len(results) and total_context_length + len(results[i][0].page_content) + len(f"--- Context\n\n") <= available_space:
        context_text += f"--- Context\n\n{results[i][0].page_content}\n\n"
        total_context_length += len(results[i][0].page_content) + len(f"--- Context\n\n")
        i += 1

    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query)
    return prompt
