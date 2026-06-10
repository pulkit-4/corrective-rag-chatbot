import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import StateGraph, END

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "rag_documents"

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    rewrite_count: int

# ── LLM & Retriever ────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Node 1: Retrieve ───────────────────────────────────────────────────────
# Fetches top 3 relevant chunks from Qdrant based on the question

def retrieve(state: GraphState) -> GraphState:
    print("--- NODE: RETRIEVE ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"question": question, "documents": documents, "rewrite_count": state.get("rewrite_count", 0), "generation": ""}


# ── Node 2: Grade Documents ────────────────────────────────────────────────
# LLM checks each retrieved chunk — is it actually relevant to the question?
# Filters out noise so we only generate from good context

def grade_documents(state: GraphState) -> GraphState:
    print("--- NODE: GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader checking if a document is relevant to a question.
        Answer only 'yes' or 'no'. No explanation.
        'yes' means the document contains information useful to answer the question.
        'no' means the document is unrelated."""),
        ("human", "Question: {question}\n\nDocument: {document}")
    ])

    grader = grade_prompt | llm

    filtered = []
    for doc in documents:
        result = grader.invoke({
            "question": question,
            "document": doc.page_content
        })
        if "yes" in result.content.lower():
            filtered.append(doc)
            print(f"  ✓ Relevant chunk kept")
        else:
            print(f"  ✗ Irrelevant chunk removed")

    return {"question": question, "documents": filtered, "rewrite_count": state["rewrite_count"], "generation": ""}

# ── Node 3: Generate ───────────────────────────────────────────────────────
# LLM generates an answer using only the graded, relevant chunks

def generate(state: GraphState) -> GraphState:
    print("--- NODE: GENERATE ---")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join([doc.page_content for doc in documents])

    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant answering questions based only on the provided context.
        If the context does not contain enough information, say 'I don't have enough information to answer this.'
        Do not make up anything outside the context."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    chain = generate_prompt | llm
    result = chain.invoke({"context": context, "question": question})

    return {"question": question, "documents": documents, "generation": result.content, "rewrite_count": state["rewrite_count"]}


# ── Node 4: Check Hallucination ────────────────────────────────────────────
# LLM checks if the generated answer is grounded in the retrieved documents
# or if it made something up

def check_hallucination(state: GraphState) -> GraphState:
    print("--- NODE: CHECK HALLUCINATION ---")
    documents = state["documents"]
    generation = state["generation"]

    context = "\n\n".join([doc.page_content for doc in documents])

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are checking if an answer is grounded in the provided context.
        Answer only 'yes' or 'no'. No explanation.
        'yes' means the answer is fully supported by the context.
        'no' means the answer contains information not found in the context."""),
        ("human", "Context:\n{context}\n\nAnswer: {generation}")
    ])

    checker = hallucination_prompt | llm
    result = checker.invoke({"context": context, "generation": generation})

    is_grounded = "yes" in result.content.lower()
    print(f"  Grounded: {is_grounded}")

    return {
        "question": state["question"],
        "documents": documents,
        "generation": generation if is_grounded else "",
        "rewrite_count": state["rewrite_count"]
    }


# ── Node 5: Rewrite Query ──────────────────────────────────────────────────
# If documents were irrelevant or answer was hallucinated,
# LLM rewrites the question to try a better retrieval

def rewrite_query(state: GraphState) -> GraphState:
    print("--- NODE: REWRITE QUERY ---")
    question = state["question"]
    rewrite_count = state["rewrite_count"]

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are rewriting a search query to be more specific and effective.
        Return only the rewritten query. Nothing else."""),
        ("human", "Original query: {question}\nRewrite it to retrieve better information.")
    ])

    rewriter = rewrite_prompt | llm
    result = rewriter.invoke({"question": question})
    new_question = result.content.strip()
    print(f"  Rewritten: {new_question}")

    return {
        "question": new_question,
        "documents": [],
        "generation": "",
        "rewrite_count": rewrite_count + 1
    }
    
# ── Conditional Edge Functions ─────────────────────────────────────────────
# These functions decide which node to go to next based on current state
 
def decide_after_grading(state: GraphState) -> str:
    """After grading docs: if relevant docs exist → generate, else → rewrite"""
    if len(state["documents"]) > 0:
        print("--- EDGE: docs relevant → generate ---")
        return "generate"
    else:
        if state["rewrite_count"] >= 2:
            print("--- EDGE: max rewrites reached → end ---")
            return "end"
        print("--- EDGE: no relevant docs → rewrite ---")
        return "rewrite_query"
 
 
def decide_after_hallucination_check(state: GraphState) -> str:
    """After hallucination check: if grounded → end, else → rewrite"""
    if state["generation"] != "":
        print("--- EDGE: answer grounded → end ---")
        return "end"
    else:
        if state["rewrite_count"] >= 2:
            print("--- EDGE: max rewrites reached → end ---")
            return "end"
        print("--- EDGE: hallucination detected → rewrite ---")
        return "rewrite_query"
 
 
# ── Build the Graph ────────────────────────────────────────────────────────
 
def build_graph():
    graph = StateGraph(GraphState)
 
    # Add all 5 nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("rewrite_query", rewrite_query)
 
    # Entry point
    graph.set_entry_point("retrieve")
 
    # Fixed edges (always go this way)
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("generate", "check_hallucination")
    graph.add_edge("rewrite_query", "retrieve")  # after rewrite, retrieve again
 
    # Conditional edges (decision points)
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "end": END
        }
    )
 
    graph.add_conditional_edges(
        "check_hallucination",
        decide_after_hallucination_check,
        {
            "end": END,
            "rewrite_query": "rewrite_query"
        }
    )
 
    return graph.compile()
 
 
# Export compiled graph
rag_graph = build_graph()