# Corrective RAG Chatbot

AI-powered Corrective RAG (CRAG) chatbot built using LangGraph, LangChain, Qdrant, MiniLM embeddings, and Llama 3.1.

## Features

* Retrieval-Augmented Generation (RAG)
* Document relevance grading
* Hallucination detection
* Query rewriting
* PDF document ingestion
* Qdrant vector database integration

## Tech Stack

* Python
* LangChain
* LangGraph
* Qdrant
* HuggingFace Embeddings (MiniLM)
* Groq Llama 3.1

## Workflow

Retrieve → Grade Documents → Generate Answer → Check Hallucination → Rewrite Query (if needed)

## Project Structure

* `graph.py` – LangGraph CRAG workflow
* `ingest.py` – PDF ingestion and vector store creation
* `main.py` – Chat interface

## Use Cases

* Enterprise knowledge base assistants
* Document question answering
* Internal support chatbots
* Retrieval-augmented AI systems
