import streamlit as st
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
from projectfns import generate_llama_embedding, LlamaEmbeddings, extract_text, add_excel_to_faiss, final_df, create_prompt, llama_answer, get_docs

token_key = ! cat ~/.keys/huggingface
login(token = token_key)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm = Ollama(model="llama3.1")


excel_files = ["bakery_income_sales_data_2022.xlsx", 
               "bakery_income_sales_data_2023.xlsx", 
               "bakery_income_sales_data_2024.xlsx",
               "bakery_operations_data_2022.xlsx", 
               "bakery_operations_data_2023.xlsx",
               "bakery_operations_data_2024.xlsx",
               "bakery_suppliers_data_2022.xlsx",
              "bakery_suppliers_data_2023.xlsx",
              "bakery_suppliers_data_2024.xlsx",
               "bakery_wages_data_2022.xlsx",
              "bakery_wages_data_2023.xlsx",
              "bakery_wages_data_2024.xlsx"]


st.write("# FinSight")
prompt = st.text_input(" ### What would you like to know?")
if st.button("Analyze data"):
    docs = get_docs(prompt, excel_files)
    docs = final_df(docs)
    final_prompt = create_prompt(docs, prompt)
    response = llama_answer(final_prompt)
    st.write("## Response: ")
    for i in response.split("\n"):
        if i != '' and i!= "Final Answer:" :
            st.markdown(i)
