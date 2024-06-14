from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from pdfminer.high_level import extract_text
from datasets import load_dataset
import model.utils as utils
import os
import glob
import pandas as pd
import json

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5" # The model used to convert text to embeddings
PDF_PATH = "./model/documents" # Directory containing all the pdf files
DB_PATH = "./model/documents/db4" # Where the database is persisted

CHUNK_SIZE = 900 # The maximum chunk size
CHUNK_OVERLAP = 80 # The overlap between chunks

# File containing the list of already processed pdfs. This is useful to prevent adding the same file twice in the db
PROCESSED_PDFS = os.path.join(DB_PATH, "processed_pdfs.json")

def add_pdf(pdf_path, splitter, db) -> list[str]:
    """
    Add a pdf file to the database.
    Returns the uid of the added documents
    """
    print(f"Adding pdf {pdf_path}")

    # We use PDFMiner since it's the only pdf loader that works well
    text = extract_text(pdf_path)

    # PDFMiner places a lot of \n, so we replace all \n\n by \n and all \n by a space
    text = text.replace("\n\n", "<NEWLINE_PLACEHOLDER>")
    text = text.replace("\n", " ")
    text = text.replace("<NEWLINE_PLACEHOLDER>", "\n")

    # Create and split document
    document = [Document(page_content=text, metadata={'source': pdf_path})]
    chunks = splitter.split_documents(document)

    # Add document to database, then persist database
    added_ids = db.add_documents(chunks)
    db.save_local(DB_PATH)

    print(f"Added {len(added_ids)} chunks\n")
    return added_ids


def add_dataset(dataset_name, qst_col, answ_col, db, filter_function=None):
    """
    Add a huggingface dataset to the database. We first filter the dataset 
    using filter_function, then remove chunks that are bigger than CHUNK_SIZE.
    """
    print(f"Adding dataset {dataset_name}")

    # Load dataset from HF
    dataset = load_dataset(dataset_name)['train']
    print(f"Original dataset size: {len(dataset)}")
    if filter_function:
        dataset = dataset.filter(filter_function)

    print(f"Size after applying filter_function: {len(dataset)}")

    # Concatenate question and answer into one string with a space in between
    dataset_df = dataset.to_pandas()
    combined = dataset_df[qst_col] + " " + dataset_df[answ_col]

    # Create a mask where the length of the combined string is less than max_chars
    mask = combined.str.len() < CHUNK_SIZE

    # Only keep the rows that have less than max_chars
    dataset_df = combined[mask]
    dataset_df = dataset_df.reset_index(drop=True)
    print(f"Size after removing big contexts: {dataset_df.shape[0]}")

    # Get a list of strings to add to the database
    texts: list[str] = dataset_df.tolist()
    metadatas = [{'source': dataset_name} for _ in texts]
    db.add_texts(texts, metadatas=metadatas)
    db.save_local(DB_PATH)



def add_sft(path: str, db):
    """
    Add an SFT dataset (in json format) to the database
    """
    print(f"Adding SFT dataset {path}")
    with open(path) as f:
        dataset = json.load(f)
    
    print(f"Original dataset size: {len(dataset)}")

    added_prompts = set()  # We don't add the same prompt twice to the database
    added_len = 0  # Keep track of the number of samples we added to the database
    for i, sample in enumerate(dataset):
        sentence = sample['prompt'] + "\n\n" + sample['completion']
        if len(sentence) < 1150 and sample['prompt'] not in added_prompts:
            added_prompts.add(sample['prompt'])
            db.add_texts([sentence], metadatas=[{'source': f"{path}:{i}"}])
            added_len += 1

    db.save_local(DB_PATH)
    print(f"Total number of added samples: {added_len}")


if __name__ == "__main__":
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    # Load embedding model
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs = {'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create/Load the database
    if os.path.isdir(DB_PATH):
        print(f"Loading existing database at {DB_PATH}")
        # db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("Creating new database")
        db = FAISS.from_texts(["adslvlkj2doj029394ojsdlfkj"], embedding_model) # Initialize db with dummy text

    # Add sft datasets to the database
    add_sft("./model/data/sft_train_m1.json", db)

    # Add datasets to the database
    add_dataset("meta-math/MetaMathQA", "original_question", "response", db)
    add_dataset("camel-ai/physics", "message_1", "message_2", db, lambda x: x["topic;"] == "Electromagnetism")
    add_dataset("Programming-Language/codeagent-python", "prompt", "response", db)
    # add_dataset("elfonthefly/STEM_DPO", "prompt", "chosen", db)
    # add_dataset("microsoft/orca-math-word-problems-200k", "question", "answer", db)

    # Add sft datasets to the database
    add_sft("./model/data/sft_train_m1.json", db)

    # Add pdfs to the database
    pdf_files = glob.glob(os.path.join(PDF_PATH, '*.pdf'))
    for pdf_file in pdf_files:
        add_pdf(pdf_file, text_splitter, db)
    
    print("Finished populating database")
    print("The number of elements in the database is ", db.index.ntotal)
    print("Finished")
