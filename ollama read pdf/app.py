import fitz  
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def read_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.split('\n\n')  
embedder = SentenceTransformer('all-MiniLM-L6-v2')  

pdf_text_sections = read_pdf_text("attention is all you need.pdf")
embeddings = embedder.encode(pdf_text_sections, convert_to_tensor=False)

dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Import Ollama for answering questions
from langchain_ollama import OllamaLLM  
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize Ollama LLM with the model name
llm = OllamaLLM(model="llama3.2", callbacks=[StreamingStdOutCallbackHandler()])

# Define the prompt template for the QA task
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# Create the LangChain LLMChain
qa_chain = LLMChain(llm=llm, prompt=prompt_template)

# Define the function to retrieve relevant context for a question
def retrieve_relevant_context(question, index, text_sections, embedder, top_k=3):
    question_embedding = embedder.encode([question])[0]
    distances, indices = index.search(np.array([question_embedding]), top_k)
    return "\n\n".join([text_sections[i] for i in indices[0]])

# Define the function to answer the question
def answer_question(question):
    context = retrieve_relevant_context(question, index, pdf_text_sections, embedder)
    answer = qa_chain.run({"context": context, "question": question})
    return answer

# Example usage
question = "