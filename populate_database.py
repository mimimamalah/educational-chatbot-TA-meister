from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from datasets import load_dataset
import model.utils as utils
import os
import glob
import pandas as pd

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5" # The model used to convert text to embeddings
DATA_PATH = "./game_data" # Directory containing all the pdf files
DB_PATH = "./game_data/db" # Where the database is persisted
CHUNK_SIZE = 800 # The maximum chunk size
CHUNK_OVERLAP = 80 # The overlap between chunks

# File containing the list of already processed pdfs. This is useful to prevent adding the same file twice in the db
PROCESSED_PDFS = os.path.join(DB_PATH, "processed_pdfs.json")

def add_pdf(pdf_path, splitter, db) -> list[str]:
    """
    Add a pdf file to the database.
    Returns the uid of the added documents
    """
    # We use PDFMiner since it's the only pdf loader that works well
    document = PDFMinerLoader(pdf_path).load()

    # Split the document into chunks.
    chunks = splitter.split_documents(document)

    # Add document to database, then persist database
    added_ids = db.add_documents(chunks)
    return added_ids


def add_dataset(dataset_name, qst_col, answ_col, db, filter_function=None):
    """
    Add a huggingface dataset to the database. We first filter the dataset 
    using filter_function, then remove chunks that are bigger than CHUNK_SIZE.
    """
    print(f"Adding dataset {dataset_name}")

    # Load dataset from HF
    dataset = load_dataset(dataset_name)
    print(f"Original dataset size: {len(dataset)}")
    if filter_function:
        dataset = dataset.filter(filter_function)

    print(f"Size after applying filter_function: {len(dataset)}")

    # Concatenate question and answer into one string with a space in between
    dataset_df = dataset['train'].to_pandas()
    combined = dataset_df[qst_col] + " " + dataset_df[answ_col]

    # Create a mask where the length of the combined string is less than max_chars
    mask = combined.str.len() < CHUNK_SIZE

    # Only keep the rows that have less than max_chars
    dataset_df = combined[mask]
    dataset_df = dataset_df.reset_index(drop=True)
    print(f"Size after removing big contexts: {dataset_df.shape[0]}")

    # Get a list of strings to add to the database
    texts: list[str] = dataset_df.tolist()
    db.add_texts(texts)


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
        db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("Creating new database")
        db = FAISS.from_texts(["adslvlkj2doj029394ojsdlfkj"], embedding_model) # Initialize db with dummy text

    # Add datasets to the database
    add_dataset("meta-math/MetaMathQA", "original_question", "response", db)
    add_dataset("camel-ai/physics", "message_1", "message_2", db, lambda x: x["topic;"] == "Electromagnetism")
    add_dataset("Programming-Language/codeagent-python", "prompt", "response", db)
    add_dataset("elfonthefly/STEM_DPO", "prompt", "chosen", db)

    # Add pdfs to the database
    pdf_files = glob.glob(os.path.join(DATA_PATH, '*.pdf'))
    for pdf_file in pdf_files:
        # Add file to database
        print(f"Adding {pdf_file}")
        print("#"*30)
        added_ids = add_pdf(pdf_file, text_splitter, db)
        print(f"Added {len(added_ids)} chunks\n")
    
    print("All pdf files processed")
    print("The number of elements in the database is ", db.index.ntotal)
    print("Persisting database ....")
    db.save_local(DB_PATH)
    print("Finished")
