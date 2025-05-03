from flask import Flask, request, render_template
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

csv_path = "/Users/andrewselius/Documents/GitHub/reviews.csv" # Replace with path as necessary

topic_keywords = {
    "reliability": ["reliable", "dependable", "breakdown", "repair", "maintenance"],
    "fuel_efficiency": ["mpg", "fuel", "gas", "mileage", "consumption"],
    "performance": ["engine", "speed", "torque", "acceleration", "handling"],
    "quality": ["luxurious", "comfy", "sturdy", "dynamic", "safe"],
    "comfort": ["seating", "interior", "ride", "suspension", "noise"]
}

def load(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=["brand", "text"], inplace=True)
    df["text"] = df["text"].str.strip()
    texts = df["text"].tolist()
    brands = df["brand"].unique()
    return df, texts, brands

def chunk(df, model_name, max_tokens=50, overlap=5, prefix_brand=True):
    # Initialize tokenizer and empty list of chunks
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chunked_data = []

    # Iterate over rows, adding the brand to each chunk to maintain context
    for _, row in df.iterrows():
        brand = row["brand"]
        text = str(row["text"])
        if prefix_brand:
            text = f"{brand}: {text}"

        # Tokenize the full review
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Create overlapping chunks
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            chunked_data.append({
                "brand": brand,
                "chunk_text": chunk_text
            })

    chunked_df = pd.DataFrame(chunked_data)
    return chunked_df


def chunk_text(df_chunked):
    return df_chunked["chunk_text"].tolist()

# Tagging function for topics
def tag_chunk(chunk_text):
    matched_topics = []
    lower_text = chunk_text.lower()
    for topic, keywords in topic_keywords.items():
        if any(keyword in lower_text for keyword in keywords):
            matched_topics.append(topic)
    return matched_topics

def extract_brand(query, df_chunked):
    query_lower = query.lower()
    brands = df_chunked["brand"].tolist()
    for brand in brands:
        if brand.lower() in query_lower:
            return brand
    return None

def extract_topics(query):
    query_lower = query.lower()
    matches = []
    for topic, keywords in topic_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            matches.append(topic)
    return matches

def build_embeddings(texts, model_name):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embedder, index, embeddings

def retrieve_context(query, embedder, index, df_chunked, texts, topics, top_k=7):
    brands = df_chunked["brand"].tolist()

    # Find relevant brand and topics for user query
    detected_brand = extract_brand(query)
    detected_topics = extract_topics(query)

    filtered_entries = []

    # Iterate over chunks, filter to those that match brand and topic tags
    for i, (text, brand, chunk_topics) in enumerate(zip(texts, brands, topics)):
        brand_match = detected_brand is None or brand.lower() == detected_brand.lower()
        topic_match = not detected_topics or any(a in chunk_topics for a in detected_topics)
        if brand_match and topic_match:
            filtered_entries.append((i, text))

    # If the input brand is matched, reconstruct the vectors + indices and retrieve relevant documents, otherwise skip to global search
    if detected_brand:
        #print(f"Brand: {detected_brand}")
        
        # Reconstruct filtered index
        filtered_indices, filtered_texts = zip(*filtered_entries)
        filtered_vectors = index.reconstruct_n(filtered_indices[0], len(filtered_indices))

        if not filtered_entries:
            print("No entries for this topic: Trying global search")
        else:
            # Encode query
            query_vec = embedder.encode([query])

            # Build FAISS index for filtered vectors
            faiss.normalize_L2(query_vec)
            filtered_index = faiss.IndexFlatIP(filtered_vectors.shape[1])
            filtered_index.add(np.array(filtered_vectors))
            
            # Create and return list of relevant documents
            _, match_indices = filtered_index.search(np.array(query_vec), top_k = 7)
            matched = [filtered_texts[i] for i in match_indices[0]]
            return " ".join(matched)
    else:
        print("No entries for this brand: Trying global search")


    # Global search for relevant documents if brand/topic match fails

    # Encode query
    query_vec = embedder.encode([query])
    
    # Create and return list of relevant documents
    _, indices = index.search(np.array(query_vec), top_k = 7)
    matched = [texts[i] for i in indices[0]]
    return " ".join(matched)

def load_summarizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

def create_summary(context, summarizer, max_length=120):
    input_text = "summarize: " + context
    result = summarizer(input_text, max_length=max_length, min_length=30, do_sample=False)
    return result[0]["summary_text"]

def generate_answer(query, embedder, index, summarizer, texts, top_k):
    context = retrieve_context(query, embedder, index, texts, top_k = top_k)
    answer = create_summary(context, summarizer)
    return answer

def chat(model_name, embedder_name, chunked_texts):
    summarizer = load_summarizer(model_name)
    embedder, index, _ = build_embeddings(chunked_texts, embedder_name)
    
    print("Auto Review Summarizer\nType 'exit' to quit.\n")
    

    while True:
        print("Available brands to ask about: Honda, Toyota, Subaru, Jeep, BMW, Mazda, Tesla, Dodge, Ford, Chevrolet")
        user_input = input("Ask about a car brand (e.g., 'Is Honda reliable?'): ").strip()
        print(f"\nUser: {user_input}\n")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = generate_answer(user_input, embedder, index, summarizer, chunked_texts, top_k=7)
        print(f"\nSummary:\n{response}\n")

def preprocess_pipeline(df, texts, df_brands, model_name, embedder_name):
    chunked_df = chunk(df, model_name)
    chunked_text = chunk_text(chunked_df)
    #matched_topics = tag_chunk(chunked_text)
    embedder, index, embeddings = build_embeddings(embedder_name)
    return embedder, index, embeddings, chunked_text
    

