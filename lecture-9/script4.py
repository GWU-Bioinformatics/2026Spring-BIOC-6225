import json
import math
import requests

corpus_of_documents = [
    "Take a leisurely walk in the park and enjoy the fresh air.",
    "Visit a local museum and discover something new.",
    "Attend a live music concert and feel the rhythm.",
    "Go for a hike and admire the natural scenery.",
    "Have a picnic with friends and share some laughs.",
    "Explore a new cuisine by dining at an ethnic restaurant.",
    "Take a yoga class and stretch your body and mind.",
    "Join a local sports league and enjoy some friendly competition.",
    "Attend a workshop or lecture on a topic you're interested in.",
    "Visit an amusement park and ride the roller coasters."
]

OLLAMA_BASE_URL = "http://localhost:11434/api"
EMBED_MODEL = "embeddinggemma"
GEN_MODEL = "llama3"


def get_embedding(text, model=EMBED_MODEL):
    url = f"{OLLAMA_BASE_URL}/embed"
    payload = {
        "model": model,
        "input": text
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    # Ollama returns embeddings in a list under "embeddings"
    return data["embeddings"][0]


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


def retrieve_top_k(query, corpus, k=3, score_threshold=0.20):
    query_embedding = get_embedding(query)

    scored_documents = []
    for doc in corpus:
        doc_embedding = get_embedding(doc)
        score = cosine_similarity(query_embedding, doc_embedding)

        if score >= score_threshold:
            scored_documents.append((doc, score))

    scored_documents.sort(key=lambda item: item[1], reverse=True)
    return scored_documents[:k]


def build_prompt(user_input, retrieved_docs):
    if not retrieved_docs:
        context_block = "No relevant context was retrieved."
    else:
        context_lines = [
            f"{idx + 1}. {doc}"
            for idx, (doc, score) in enumerate(retrieved_docs)
        ]
        context_block = "\n".join(context_lines)

    prompt = f"""
You are a bot that makes recommendations for activities.
Answer in very short sentences and do not include extra information.

User input:
{user_input}

Retrieved context:
{context_block}

Instructions:
- Base your answer on the retrieved context.
- If the retrieved context is not relevant enough, say you need more detail.
- Do not mention retrieval, documents, or scores.
"""
    return prompt.strip()


def generate_response(prompt, model=GEN_MODEL):
    url = f"{OLLAMA_BASE_URL}/generate"
    payload = {
        "model": model,
        "prompt": prompt
    }

    full_response = []
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()

    try:
        for line in response.iter_lines():
            if line:
                decoded_line = json.loads(line.decode("utf-8"))
                full_response.append(decoded_line.get("response", ""))
    finally:
        response.close()

    return "".join(full_response)


if __name__ == "__main__":
    user_input = "I want something outdoors and active"

    top_docs = retrieve_top_k(
        query=user_input,
        corpus=corpus_of_documents,
        k=3,
        score_threshold=0.20
    )

    print("Top retrieved documents:")
    if top_docs:
        for rank, (doc, score) in enumerate(top_docs, start=1):
            print(f"{rank}. score={score:.3f} | {doc}")
    else:
        print("No documents passed the score threshold.")

    prompt = build_prompt(user_input, top_docs)

    print("\nGenerated response:")
    try:
        answer = generate_response(prompt, model=GEN_MODEL)
        print(answer)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
