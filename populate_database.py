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

# File containing the list of already processed pdfs. This is useful to prevent adding the same file twice in the db
PROCESSED_PDFS = os.path.join(DB_PATH, "processed_pdfs.json")

def add_pdf(pdf_path, splitter, db) -> list[str]:
    """
    Add a pdf file to the database, then persist the database.
    Returns the uid of the added documents
    """

    # We use PDFMiner since it's the only pdf loader that works well
    document = PDFMinerLoader(pdf_path).load()

    # Split the document into chunks.
    chunks = splitter.split_documents(document)

    # Add document to database, then persist database
    added_ids = db.add_documents(chunks)
    db.save_local(DB_PATH)
    return added_ids

def create_context(dataset, qst_col, answ_col, max_chars=800) -> pd.DataFrame:
    dataset_df = pd.DataFrame(dataset)
    # Concatenate question and answer into one string with a space in between
    combined = dataset_df[qst_col] + " " + dataset_df[answ_col]

    # Create a mask where the length of the combined string is less than max_chars
    mask = combined.str.len() < max_chars

    # only keep the rows that have less than max_chars
    dataset_df = combined[mask]
    dataset_df = dataset_df.reset_index(drop=True)

    # return the list
    return dataset_df.tolist()


if __name__ == "__main__":
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len
    )

    # Load embedding model
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs = {'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Loading datasets...")
    # Load datasets from Hugging Face
    metamath_dataset = load_dataset("meta-math/MetaMathQA")
    physics_dataset = load_dataset("camel-ai/physics")
    python_dataset = load_dataset("Programming-Language/codeagent-python")
    stem_dataset = load_dataset("elfonthefly/STEM_DPO")

    metamath_db = create_context(metamath_dataset["train"], "original_question", "response")
    physics_db = create_context(physics_dataset["train"], "message_1", "message_2")
    python_db = create_context(python_dataset["train"], "prompt", "answer")
    stem_db = create_context(stem_dataset["train"], "prompt", "chosen")

    print("Datasets loaded")

    
    # Create/Load the database
    if os.path.isdir(DB_PATH):
        print(f"Loading existing database at {DB_PATH}")
        db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        already_added = utils.read_json(PROCESSED_PDFS)
    else:
        print("Creating new database")
        db = FAISS.from_texts(["adslvlkj2doj029394ojsdlfkj"], embedding_model) # Initialize db with dummy text
        already_added = []
        

    print("Adding external datasets to database...")
    db.add_texts(metamath_db)
    db.add_texts(physics_db)
    db.add_texts(python_db)
    db.add_texts(stem_db)
    db.save_local(DB_PATH)
    print("External datasets added to database\n")

    # Process all the pdf files in the DATA_PATH directory
    pdf_files = glob.glob(os.path.join(DATA_PATH, '*.pdf'))
    for pdf_file in pdf_files:
        if pdf_file in already_added:
            print(f"Skipping {pdf_file} as it has already been added")
            continue

        # Add file to database
        print(f"Adding {pdf_file}")
        print("#"*30)
        added_ids = add_pdf(pdf_file, text_splitter, db)
        print(f"Added {len(added_ids)} chunks\n")

        # Update the list of already added pdfs
        already_added.append(pdf_file)
        utils.write_json(already_added, PROCESSED_PDFS)
    
    print("All pdf files processed")

    print("The number of elements in the database is ", db.index.ntotal)