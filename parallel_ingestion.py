import faiss
import numpy as np
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load the SentenceTransformer embeddings model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# FAISS Index (flat index, no special optimizations)
faiss_index = faiss.IndexFlatL2(embedding_model.dimension)

# FAISS Vector Store
faiss_store = FAISS(embedding_model.embed_query, faiss_index, None)

# Storage for text chunks (used to map FAISS vector index back to original chunks)
text_chunks = []

# Ingestion Function: Loads and stores documents in FAISS
def ingest_documents(file_paths):
    """
    Ingest documents from file paths, split them into chunks, and store them in FAISS for fast retrieval.
    
    :param file_paths: List of file paths to load documents from
    """
    global text_chunks
    
    for file_path in file_paths:
        # Step 1: Load documents
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Step 2: Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        # Step 3: Generate embeddings for each chunk and store them in FAISS
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = embedding_model.embed_documents(chunk_texts)
        
        # Add chunks and embeddings to FAISS and storage
        faiss_store.add_documents(chunk_embeddings)
        text_chunks.extend(chunk_texts)  # Store original text chunks for retrieval

    print(f"Successfully ingested {len(text_chunks)} text chunks into FAISS.")

# Query Function: Uses FAISS to retrieve relevant chunks and generate an augmented answer
def query_pipeline(query, top_k=3):
    """
    Query FAISS for the top-k relevant document chunks, then generate an answer using those chunks.
    
    :param query: The input query string
    :param top_k: Number of top relevant document chunks to retrieve
    :return: Augmented answer based on retrieved chunks
    """
    # Step 1: Generate embedding for the query
    query_embedding = embedding_model.embed_query(query)
    
    # Step 2: Perform similarity search in FAISS
    D, I = faiss_store.index.search(np.array([query_embedding]), top_k)
    
    # Step 3: Retrieve the most relevant text chunks
    retrieved_chunks = [text_chunks[i] for i in I[0] if i < len(text_chunks)]
    
    # Step 4: Use a generative model to augment the answer based on the retrieved chunks
    llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo")  # This could be any LLM
    retriever = RetrievalQA.from_chain_type(llm, retriever=faiss_store.as_retriever())
    
    # Combine the query with relevant chunks for answer generation
    answer = retriever.run(query)
    
    return answer

# Example Usage
if _name_ == "_main_":
    # Ingest example documents (can be paths to text files)
    example_file_paths = [
        "/path/to/document1.txt",
        "/path/to/document2.txt"
    ]
    ingest_documents(example_file_paths)
    
    # Query the pipeline
    query = "What is the Eiffel Tower?"
    response = query_pipeline(query)
    
    print(response)