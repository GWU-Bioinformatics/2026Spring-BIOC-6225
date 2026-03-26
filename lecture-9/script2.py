import json
import math
import re
from collections import Counter
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

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "to", "was", "were", "will", "with", "you", "your"
}


def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def compute_idf(corpus):
    num_docs = len(corpus)
    document_frequency = Counter()

    for doc in corpus:
        unique_tokens = set(tokenize(doc))
        for token in unique_tokens:
            document_frequency[token] += 1

    idf = {}
    for token, df in document_frequency.items():
        idf[token] = math.log((num_docs + 1) / (df + 1)) + 1

    return idf


def compute_tfidf_vector(text, idf):
    tokens = tokenize(text)
    if not tokens:
        return {}

    term_counts = Counter(tokens)
    total_terms = len(tokens)

    vector = {}
    for term, count in term_counts.items():
        tf = count / total_terms

        term_idf = idf.get(term, math.log((len(corpus_of_documents) + 1) / 1) + 1)
        vector[term] = tf * term_idf

    return vector


def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0

    common_terms = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)

    norm1 = math.sqrt(sum(weight * weight for weight in vec1.values()))
    norm2 = math.sqrt(sum(weight * weight for weight in vec2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


def retrieve_top_k(query, corpus, k=3, score_threshold=0.10):
    idf = compute_idf(corpus)
    query_vector = compute_tfidf_vector(query, idf)

    scored_documents = []
    for doc in corpus:
        doc_vector = compute_tfidf_vector(doc, idf)
        score = cosine_similarity(query_vector, doc_vector)
        if score >= score_threshold:
            scored_documents.append((doc, score))

    scored_documents.sort(key=lambda item: item[1], reverse=True)
    return scored_documents[:k]


def build_prompt(user_input, retrieved_docs):
    if not retrieved_docs:
        retrieved_text = "No relevant activities were retrieved."
    else:
        retrieved_lines = [
            f"- {doc} (score={score:.3f})"
            for doc, score in retrieved_docs
        ]
        retrieved_text = "\n".join(retrieved_lines)

    prompt = f"""
You are a bot that makes recommendations for activities.
You answer in very short sentences and do not include extra information.

The user input is: {user_input}

Retrieved activities:
{retrieved_text}

If relevant activities were retrieved, recommend the best one.
If nothing relevant was retrieved, say that you need more detail from the user.
"""
    return prompt.strip()


def generate_response(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt
    }
    headers = {"Content-Type": "application/json"}

    full_response = []

    response = requests.post(
        url,
        data=json.dumps(data),
        headers=headers,
        stream=True
    )

    try:
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = json.loads(line.decode("utf-8"))
                full_response.append(decoded_line.get("response", ""))
    finally:
        response.close()

    return "".join(full_response)



if __name__ == "__main__":
    user_input = "I like to hike"

    top_docs = retrieve_top_k(
        query=user_input,
        corpus=corpus_of_documents,
        k=3,
        score_threshold=0.10
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
        model_output = generate_response(prompt, model="llama3")
        print(model_output)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
