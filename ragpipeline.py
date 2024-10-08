import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load a transformer model to create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage for embeddings and documents
document_store = []

# Ingestion function: Store documents and their embeddings
def ingest_documents(documents):
    """
    Ingest a list of documents and store their embeddings in memory.
    
    :param documents: List of documents (strings) to ingest
    """
    global document_store
    for doc in documents:
        # Create an embedding for the document
        embedding = model.encode(doc)
        # Store the document and its embedding
        document_store.append({'text': doc, 'embedding': embedding})
    print(f"Successfully ingested {len(documents)} documents.")

# Query function: Retrieve and generate a response from relevant documents
def query_pipeline(query, top_k=3):
    """
    Retrieve relevant documents from the document store and return an augmented answer.
    
    :param query: The input query string
    :param top_k: Number of top relevant documents to retrieve
    :return: Augmented answer based on retrieved documents
    """
    # Generate embedding for the query
    query_embedding = model.encode(query)
    
    # Calculate cosine similarity between query and documents
    similarities = []
    for doc in document_store:
        sim = cosine_similarity([query_embedding], [doc['embedding']])[0][0]
        similarities.append(sim)
    
    # Find top-k most similar documents
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_docs = [document_store[i]['text'] for i in top_k_indices]
    
    # Generate an augmented answer based on the retrieved documents
    answer = generate_answer_from_docs(query, relevant_docs)
    
    return answer

# Simple function to generate an answer based on retrieved docs (mock function)
def generate_answer_from_docs(query, documents):
    """
    Generate an answer based on the query and retrieved documents.
    
    :param query: The query string
    :param documents: List of retrieved documents
    :return: Generated answer string
    """
    # In a real scenario, this would involve a generative model like GPT
    # Here, we'll just concatenate the top documents as a mock response
    response = f"Answer based on the following documents:\n\n" + "\n---\n".join(documents)
    return response

# Example usage
if _name_ == "_main_":
    # Ingest example documents
    example_docs = [
        "The Eiffel Tower is located in Paris.",
        "Mount Everest is the highest mountain in the world.",
        "Python is a popular programming language for AI development."
    ]
    ingest_documents(example_docs)
    
    # Query the pipeline
    query = "Where is the Eiffel Tower located?"
    response = query_pipeline(query)
    
    print(response)