from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
import torch
import re

QUERY = "How much money is given to the player at the start of a Monopoly game?"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" # The model used to convert text to embeddings
LLM_MODEL = "PeterAM4/EPFL-TA-Meister"
DB_PATH = "./game_data/db" # Where the database is persisted

PROMPT_TEMPLATE = """
Answer the question. You can utilize the following context:

{context}

---

Answer the question. You may use the above context:
{question}

---

Remember that this is a multiple-choice question. The response you should give is a single letter. Give an explanation to the answer then respond with "The correct answer is : (letter)"
Explanation:
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # Load llama3
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL).to(device)

    # Load embedding model
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs = {'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Prepare the DB.
    db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)

    # Search the DB.
    results = db.similarity_search_with_score(QUERY, k=2)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=QUERY)
    print("Prompt:")
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    response_ids = model.generate(**inputs)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=False)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

