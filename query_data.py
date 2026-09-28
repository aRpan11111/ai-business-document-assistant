import argparse

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text):
    """
    Query the Chroma vector database and generate
    an answer using OpenAI.
    """

    # Prepare the embedding function
    embedding_function = OpenAIEmbeddings()

    # Load the Chroma database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Search the database
    results = db.similarity_search_with_relevance_scores(
        query_text,
        k=3
    )

    # Check whether relevant documents were found
    if len(results) == 0 or results[0][1] < 0.7:
        return "Unable to find matching results."

    # Build context from retrieved documents
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results]
    )

    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    prompt = prompt_template.invoke({
        "context": context_text,
        "question": query_text
    })

    # Create LLM
    model = ChatOpenAI()

    # Generate response
    response = model.invoke(prompt)

    # Extract response text
    response_text = response.content

    # Get source documents
    sources = [
        doc.metadata.get("source", None)
        for doc, _score in results
    ]

    formatted_response = (
        f"Response: {response_text}\n\n"
        f"Sources: {sources}"
    )

    return formatted_response


def main():
    # Create CLI
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query_text",
        type=str,
        help="The query text."
    )

    args = parser.parse_args()

    query_text = args.query_text

    # Query RAG
    response = query_rag(query_text)

    print(response)


if __name__ == "__main__":
    main()