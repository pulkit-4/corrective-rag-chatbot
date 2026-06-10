from graph import rag_graph

def chat():
    print("\n=== RAG Chatbot (CRAG) ===")
    print("Type 'exit' to quit\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        print()
        result = rag_graph.invoke({
            "question": question,
            "documents": [],
            "generation": "",
            "rewrite_count": 0
        })

        print("\n--- FINAL ANSWER ---")
        if result["generation"]:
            print(result["generation"])
        else:
            print("Sorry, I could not find a reliable answer in the document.")
        print()

if __name__ == "__main__":
    chat()