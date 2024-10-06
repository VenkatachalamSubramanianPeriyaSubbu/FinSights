import pandas as pd
import numpy as np
import llama_index
import transformers
import torch
import faiss
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

def generate_llama_embedding(text):
    return embed_model.get_text_embedding(text)
    
class LlamaEmbeddings:
    def embed_query(self, text):
        return generate_llama_embedding(text)  

login(token="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")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm = Ollama(model="llama3.1")



def extract_text(path):
    """Extract text content from all sheets of an Excel file."""
    data = pd.read_excel(path)
    extracted_text = ""
    for row in data.iterrows():
        extracted_text += row[1].to_string()
       
    return extracted_text

def add_excel_to_faiss(file_name, index, vector_store):
    text = extract_text("Bakery Data/"+file_name)
    embedding = LlamaEmbeddings().embed_query(text)
    index.add(np.array([embedding]))
    doc_id = len(vector_store.index_to_docstore_id)  
    vector_store.index_to_docstore_id[doc_id] = str(doc_id)
    vector_store.docstore.add({str(doc_id):"Bakery Data/"+file_name})
    

def final_df(docs):
    return [pd.read_excel(doc) for doc in docs]
    
def get_docs(user_prompt, excel_files):
    embedding_dim = len(generate_llama_embedding("My Project"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
    embedding_function=LlamaEmbeddings(),
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
    )
    for excel in excel_files:
        add_excel_to_faiss(excel, index, vector_store)
    user_prompt_embedding = np.array([embed_model.get_text_embedding(user_prompt)])
    k = 3
    distances, indices = vector_store.index.search(user_prompt_embedding, k)
    retrieved_docs = []
    
    for idx in indices[0]:
        if idx in vector_store.index_to_docstore_id:
            doc_id = vector_store.index_to_docstore_id[idx]
            retrieved_docs.append(vector_store.docstore.search(doc_id))
    
    return retrieved_docs

def create_prompt(retrieved_docs, user_prompt):

    template = """
    You are an expert financial analyst.
    given my bakery shop data : {financial_data},
    {user_prompt}. Remember that, wages, operations, money paid to suppliers are expenses and income represents the money from sale of bakery items. Do not give me code. Just give me final answer and then the recommendations or thoughts you have as a expert financial
    analyst.
    """

    prompt = PromptTemplate(
        input_variables=["financial_data", "user_prompt"],
        template=template
    )


    inputs = {
        "financial_data": retrieved_docs,
        "user_prompt":user_prompt
    }
    formatted_prompt = prompt.format(**inputs)
    return formatted_prompt

def llama_answer(prompt):
    response = llm.invoke(prompt)
    return response



