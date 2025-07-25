from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
from sentence_transformers import CrossEncoder
from llama_cpp import Llama
from spellchecker import SpellChecker

import re
import os
import pandas as pd
import csv
from datetime import datetime

# CONFIG

DATA_DIR = "data/"
CSV_FILE = os.path.join(DATA_DIR, "disease_symptoms.csv")
PDF_DIR = DATA_DIR
DB_FAISS_PATH = "vectorstore/db_faiss"

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
LOG_FILE = "query_log.csv"


spell = SpellChecker()

COMMON_FIXES = {
    "join pain": "joint pain",
    "headacghe": "headache",
    "fevar": "fever",
    "throat pane": "throat pain",
    "stomache": "stomach",
    "vomitting": "vomiting"
}

MEDICAL_SYNONYMS = {
    "fever": ["pyrexia", "high temperature"],
    "joint pain": ["arthralgia"],
    "throat pain": ["sore throat", "pharyngitis"],
    "headache": ["cephalalgia"],
    "cough": ["tussis"],
    "fatigue": ["tiredness", "lethargy", "exhaustion"],
    "rash": ["eruption", "exanthem"],
    "nausea": ["sickness", "queasiness"],
    "diarrhea": ["loose stools"],
    "constipation": ["irregularity"]
}

def correct_spelling(query):
    words = re.findall(r'\w+', query.lower())
    corrected = [spell.correction(w) if spell.unknown([w]) else w for w in words]
    return " ".join([w for w in corrected if w is not None])

def apply_common_fixes(query):
    for wrong, correct in COMMON_FIXES.items():
        query = query.replace(wrong, correct)
    return query

def expand_synonyms(query):
    new_terms = []
    terms = re.findall(r'\w+', query.lower())
    for term in terms:
        for key, synonyms in MEDICAL_SYNONYMS.items():
            if term in key or any(term in syn for syn in synonyms):
                new_terms.extend(synonyms)
    if new_terms:
        return query + " " + " ".join(set(new_terms))
    return query

def load_disease_csv(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    symptom_cols = [c for c in df.columns if c != "prognosis"]
    df['symptoms'] = df.apply(
        lambda row: [col.replace('_',' ').strip() for col in symptom_cols if row[col] == 1],
        axis=1
    )
    return df

def match_diseases(query, df):
    if df is None:
        return []
    query_terms = set(re.findall(r'\w+', query.lower()))
    match_scores = {}
    for i, row in df.iterrows():
        symptoms = row['symptoms']
        symptom_terms = set()
        for s in symptoms:
            symptom_terms.update(re.findall(r'\w+', s.lower()))
        overlap = query_terms & symptom_terms
        score = len(overlap)
        if overlap:
            match_scores[row['prognosis']] = score
    return sorted(match_scores.keys(), key=lambda x: match_scores[x], reverse=True)

# LangChain VectorStore Pipeline

def build_langchain_vectorstore():
    # Load pre-processed chunks from JSON
    import json
    from langchain.schema import Document
    
    chunks = []
    try:
        with open(os.path.join(DATA_DIR, "merck_chunks.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle different JSON structures
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = item.get("text", item.get("content", ""))
                        metadata = item.get("metadata", {})
                        if text:
                            chunks.append(Document(page_content=text, metadata=metadata))
                    elif isinstance(item, str):
                        chunks.append(Document(page_content=item, metadata={}))
            elif isinstance(data, dict):
                # If it's a single document
                text = data.get("text", data.get("content", ""))
                if text:
                    chunks.append(Document(page_content=text, metadata=data.get("metadata", {})))
    except Exception as e:
        print(f"Error loading chunks: {e}")
        # Create a simple fallback document
        chunks = [Document(page_content="Medical information not available.", metadata={})]

    # Build embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(chunks, embedding_model)
    return db, embedding_model

def load_langchain_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db, embedding_model

# LLM Generation

llm = None

def get_llm_response(query, context, disease_names=None):
    global llm
    if llm is None:
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=8192,
            n_threads=4,
            n_gpu_layers=-1 if torch.cuda.is_available() else 0,
            verbose=False
        )
    
    if disease_names:
        prompt = f"""
You are a medical assistant. Provide precise, accurate medical information based on the context.

Context from medical reference:
{context}

User symptoms: {query}

Provide a concise response (max 1000 tokens) in this format:

POTENTIAL DISEASES:
[List 2-3 most likely diseases with brief descriptions]

RECOMMENDATIONS:
[Specific, actionable medical advice]


Keep information precise and evidence-based.
"""
    else:
        prompt = f"""
You are a medical assistant. Provide precise, accurate medical information based on the context.

Context:
{context}

User symptoms: {query}

Provide a concise response (max 1000 tokens) in this format:

RECOMMENDATIONS:
[Specific, actionable medical advice]


Keep information precise and evidence-based.
"""
    # llama_cpp completion call
    completion = llm.create_completion(
        prompt=prompt,
        max_tokens=200,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.1,
    )
    # Handle the completion response properly
    if isinstance(completion, dict) and "choices" in completion and completion["choices"]:
        response = completion["choices"][0]["text"].strip()
        return response + "\n\nDisclaimer: This is not medical advice. Please consult a doctor."
    else:
        return "Unable to generate response.\n\nDisclaimer: This is not medical advice. Please consult a doctor."

# Logging

def log_query(query, result_length):
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not log_exists:
            writer.writerow(["timestamp", "query", "result_length"])
        writer.writerow([datetime.now().isoformat(), query, result_length])

# Processing pipeline

def process_query(query, disease_df, db, embedding_model, return_response=False):
    query_original = query

    # Preprocessing
    query = apply_common_fixes(query)
    query = correct_spelling(query)
    expanded_query = expand_synonyms(query)

    # Disease matching
    matched_diseases = match_diseases(query, disease_df)

    # Retrieve contexts
    disease_contexts = []
    if matched_diseases:
        disease_queries = matched_diseases[:5]
        disease_context_results = []
        for dq in disease_queries:
            disease_context_results.extend(db.similarity_search(dq, k=3))
        disease_contexts = list(set([d.page_content for d in disease_context_results]))

    # Semantic + cross-encoder rerank
    semantic_results = db.similarity_search(expanded_query, k=5)
    semantic_contexts = [d.page_content for d in semantic_results]

    ranked_semantic_contexts = []
    if semantic_contexts:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu')
        pairs = [[query_original, doc] for doc in semantic_contexts]
        scores = cross_encoder.predict(pairs)
        ranked_semantic_contexts = [
            doc for _, doc in sorted(zip(scores, semantic_contexts), key=lambda x: x[0], reverse=True)
        ]

    # Compose final prompt
    full_context = "\n\n".join(disease_contexts + ranked_semantic_contexts[:3])
    response = get_llm_response(query_original, full_context, matched_diseases[:3])

    log_query(query_original, len(response))

    if return_response:
        return response
    else:
        # Optional: Console output for testing
        print("MEDICAL ANALYSIS RESULTS")
        if matched_diseases:
            print(f"\n🔍 POTENTIAL DISEASES:")
            for i, disease in enumerate(matched_diseases[:3], 1):
                print(f"   {i}. {disease}")
        print(f"\n📋 AI RECOMMENDATIONS:")
        print(response)

if __name__ == "__main__":
    print("Initializing system...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using: ", device)
    # Load or build vector DB
    if os.path.exists(f"{DB_FAISS_PATH}/index.faiss"):
        db, embedding_model = load_langchain_vectorstore()
        print("Vectorstore loaded.")
    else:
        db, embedding_model = build_langchain_vectorstore()
        print("Vectorstore built and saved.")

    disease_df = load_disease_csv(CSV_FILE)

    print("System ready. Enter your symptoms (or 'quit'):")

    while True:
        try:
            query = input("\nSymptoms: ").strip()
            if query.lower() in ("quit", "exit"):
                break
            if query:
                process_query(query, disease_df, db, embedding_model)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)