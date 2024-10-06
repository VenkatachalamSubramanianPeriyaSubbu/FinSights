# FinSights
 AI Financial Analysis tool for micro and small business using LLMs

## Overview
 This project leverages a suite of powerful tools and libraries—Pandas, Hugging Face, FAISS, Ollama, Llama 3.1, and Lang Chain—to provide comprehensive data analysis and natural language
 processing solutions. By integrating Retrieval Augmented Generation (RAG), LLM and effective data handling and visualization capabilities, this project aims to offer enhanced insights and automation.

## Features
 1. Data Manipulation with Pandas: Efficient data cleaning, transformation, and analysis.
 2. Natural Language Processing (NLP) with Hugging Face: Utilizes embedding system "BAAI/bge-small-en-v1.5".
 3. FAISS (Facebook AI Similarity Search) for Similarity Search: Implements a fast, scalable solution for similarity searches, suitable for recommendation systems and semantic search.
 4. Ollama and Llama 3.1: Provides cutting-edge language model that enhance the NLP capabilities of the application.
 5. Lang Chain Integration: Orchestrates large language models to facilitate complex conversations, decision-making processes, and chained tasks.

## Prerequisites
- Python 3.8+
- Required Python packages: pandas, transformers, faiss-cpu, langchain, llama

## Project Structure
`Bakery Data/`: Contains input datasets for an imaginary bakery shop.<br>
`pages/`: Python scripts for streamlit application pages.<br>
`Analysis.py`: Python script for main page of the streamlit application.<br>
`plotting.py`: Python script of visualization code.<br>
`projectfns.py`: Python script containing functions used in `Analysis.py`.<br>
`README.md`: Project documentation.

## Running the Application

To run this project, clone the repository and execute the following command:

```bash
streamlit run Analysis.py
