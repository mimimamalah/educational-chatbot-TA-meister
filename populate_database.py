from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import model.utils as utils
import os
import glob

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

    # Create/Load the database
    if os.path.isdir(DB_PATH):
        print(f"Loading existing database at {DB_PATH}")
        db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        already_added = utils.read_json(PROCESSED_PDFS)
    else:
        print("Creating new database")
        db = FAISS.from_texts(["adslvlkj2doj029394ojsdlfkj"], embedding_model) # Initialize db with dummy text
        already_added = []
        

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
