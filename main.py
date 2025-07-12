from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import spacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from autocorrect import Speller
import string
import re
import datefinder
import os
import time
from rank_bm25 import BM25Okapi
import json
from collections import defaultdict
import csv

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stemmer = PorterStemmer()
spell = Speller(lang='en')

with open("common_words", "r") as file:
    stopwords = re.findall(r"\S+", file.read())
stop_words_set = set(stopwords)

tfidf_cache = {
    "wikIR1k": {"vectorizer": None, "matrix": None, "docs": [], "filenames": [], "clusters": [], "inverted_index": None, "bm25": None, "docid_to_postid": {}},
    "lifestyle": {"vectorizer": None, "matrix": None, "docs": [], "filenames": [], "clusters": [], "inverted_index": None, "bm25": None, "docid_to_postid": {}}
}

LIFESTYLE_QUERIES = []
LIFESTYLE_RELEVANCE = {}

WIKIR_QUERIES = []  

class InvertedIndex:
    def __init__(self):
        self.index = {}
    def add_document(self, doc_id, terms):
        for term in terms:
            self.index.setdefault(term, set()).add(doc_id)

def stem_words(txt):
    return [stemmer.stem(word) for word in txt]

def expand_query(words):
    expanded = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if '_' not in lemma.name():
                    expanded.add(lemma.name().lower())
    return list(expanded)

def cos_similarity(search_query_weights, tfidf_weights_matrix):
    return cosine_similarity(search_query_weights, tfidf_weights_matrix)[0]

def most_similar(similarity_list, min_Doc=10):
    sim_list = np.array(similarity_list)
    return list(sim_list.argsort()[-min_Doc:][::-1])

def suggest_queries_from_examples(current_query, example_queries, top_k=3):
    if not example_queries:
        return []
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(example_queries + [current_query])
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    top_indices = cosine_sim.argsort()[-top_k:][::-1]
    return [example_queries[i][:80] + '...' for i in top_indices]

def cluster_documents(docs, num_clusters=5):
    tfidf = TfidfVectorizer(max_features=30000)
    tfidf_matrix = tfidf.fit_transform(docs)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(tfidf_matrix)

def load_collection_tsv(filepath):
    docs, filenames = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                doc_id, text = parts[0], parts[1]
                filenames.append(doc_id)
                docs.append(text)
    return docs, filenames

def load_queries_jsonl(filepath):
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries.append({"qid": str(obj["qid"]), "query": obj["query"]})
    return queries

def load_relevance_jsonl(filepath):
    relevance = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["qid"])  # ← المفتاح الصحيح
            post_ids = [str(pid) for pid in obj.get("answer_pids", [])]  # ← المفتاح الصحيح
            relevance[qid].extend(post_ids)
    return relevance

def load_queries_tsv(filepath):
    queries = []
    with open(filepath, newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                queries.append({"qid": row[0], "query": row[1]})
    return queries


def load_relevance_csv(filepath):
    relevance = defaultdict(list)
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            qid = row['id_left']
            docid = row['id_right']
            label = int(row['label'])
            if label > 0:
                relevance[qid].append(docid)
    return relevance

def load_wikir_queries_csv(filepath):
    queries = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            queries.append({"qid": row["id_left"], "query": row["text_left"]})
    return queries

def build_docid_to_postid_mapping(collection_docs, relevance_data):
    mapping = {}
    for doc_id in collection_docs:
        if doc_id in relevance_data:
            mapping[doc_id] = doc_id
        else:
            mapping[doc_id] = doc_id
    return mapping

def prepare_lifestyle_cache():
    global LIFESTYLE_QUERIES, LIFESTYLE_RELEVANCE

    docs, filenames = load_collection_tsv("lifestyle/dev/collection.tsv")
    inverted_index = InvertedIndex()
    table = str.maketrans('', '', string.punctuation)
    processed_docs = []

    for doc_id, text in zip(filenames, docs):
        tokens = word_tokenize(text.lower())
        stripped = [w.translate(table) for w in tokens]
        words = [w for w in stripped if w.isalpha() and w not in stop_words_set]
        stemmed = stem_words(words)
        inverted_index.add_document(doc_id, stemmed)
        processed_docs.append(' '.join(words))

    clusters = cluster_documents(processed_docs)
    vectorizer = TfidfVectorizer(max_features=50000)
    matrix = vectorizer.fit_transform(processed_docs)
    bm25 = BM25Okapi([doc.split() for doc in processed_docs])

    LIFESTYLE_QUERIES = load_queries_tsv("lifestyle/dev/questions.search.tsv")

    LIFESTYLE_RELEVANCE = load_relevance_jsonl("lifestyle/dev/qas.search.jsonl")
    # LIFESTYLE_RELEVANCE = load_relevance_jsonl("lifestyle/dev/qas.forum.jsonl")

    docid_to_postid = {}
    for qid, postids in LIFESTYLE_RELEVANCE.items():
        for pid in postids:
            docid_to_postid[pid] = pid

    tfidf_cache["lifestyle"].update({
        "vectorizer": vectorizer,
        "matrix": matrix,
        "docs": processed_docs,
        "filenames": filenames,
        "clusters": clusters,
        "inverted_index": inverted_index,
        "bm25": bm25,
        "docid_to_postid": docid_to_postid,
        "relevance_data": LIFESTYLE_RELEVANCE
    })


def prepare_wikir_cache():
    global WIKIR_QUERIES
    csv_path = "wikIR1k/documents.csv"
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path, header=None, names=["id", "text"], dtype={"id": str}, low_memory=False)
    docs, filenames = [], []
    inverted_index = InvertedIndex()
    table = str.maketrans('', '', string.punctuation)
    for doc_id, text in zip(df["id"], df["text"]):
        tokens = word_tokenize(str(text).lower())
        stripped = [w.translate(table) for w in tokens]
        words = [w for w in stripped if w.isalpha() and w not in stop_words_set]
        stemmed = stem_words(words)
        inverted_index.add_document(str(doc_id), stemmed)
        docs.append(' '.join(words))
        filenames.append(str(doc_id))
    clusters = cluster_documents(docs)
    vectorizer = TfidfVectorizer(max_features=50000)
    matrix = vectorizer.fit_transform(docs)
    bm25 = BM25Okapi([doc.split() for doc in docs])

    queries_csv_path = "wikIR1k/test/queries.csv"
    if os.path.exists(queries_csv_path):
        WIKIR_QUERIES = load_wikir_queries_csv(queries_csv_path)
    else:
        WIKIR_QUERIES = []

    qrels_csv_path = "wikIR1k/test/BM25.qrels.csv"
    if os.path.exists(qrels_csv_path):
        relevance_data = load_relevance_csv(qrels_csv_path)
    else:
        relevance_data = {}

    docid_to_postid = {}
    for qid, postids in relevance_data.items():
        for pid in postids:
            docid_to_postid[pid] = pid

    tfidf_cache["wikIR1k"] = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "docs": docs,
        "filenames": filenames,
        "clusters": clusters,
        "inverted_index": inverted_index,
        "bm25": bm25,
        "docid_to_postid": docid_to_postid,
        "relevance_data": relevance_data
    }

prepare_lifestyle_cache()
prepare_wikir_cache()

RELEVANCE_DATA = {
    "wikIR1k": tfidf_cache["wikIR1k"].get("relevance_data", {}),
    "lifestyle": LIFESTYLE_RELEVANCE
}
def retrieve_docs_from_inverted_index(index, query_terms):
    doc_candidates = set()
    for term in query_terms:
        if term in index.index:
            doc_candidates.update(index.index[term])
    return list(doc_candidates)
@app.get("/")
def run_text_processing(q: str = Query(...), dataset: str = Query("lifestyle"), method: str = Query("tfidf")):
    start_time = time.time()
    example_queries = [q["query"] for q in LIFESTYLE_QUERIES] if dataset == "lifestyle" else []

    query_spelled = spell(q)
    tokens_query = word_tokenize(query_spelled.lower())
    table = str.maketrans('', '', string.punctuation)
    stripped_query = [w.translate(table) for w in tokens_query]
    words_query = [w for w in stripped_query if w and w not in stop_words_set]
    stemmed_query = stem_words(words_query)
    expanded_words_query = expand_query(words_query)

    docs = tfidf_cache[dataset]["docs"]
    filenames = tfidf_cache[dataset]["filenames"]
    inverted_index = tfidf_cache[dataset]["inverted_index"]
    clusters = tfidf_cache[dataset]["clusters"]
    bm25 = tfidf_cache[dataset]["bm25"]
    vectorizer = tfidf_cache[dataset]["vectorizer"]
    matrix = tfidf_cache[dataset]["matrix"]
    docid_to_postid = tfidf_cache[dataset]["docid_to_postid"]
    relevance_data = RELEVANCE_DATA.get(dataset, {})

    # ✅ ترشيح أولي باستخدام الفهرس العكسي
    candidate_doc_ids = retrieve_docs_from_inverted_index(inverted_index, stemmed_query)
    if not candidate_doc_ids:
        return {"message": "لا يوجد مستندات مرشحة للاستعلام الحالي", "query": q}

    doc_idx_map = {doc_id: i for i, doc_id in enumerate(filenames)}
    candidate_indices = [doc_idx_map[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_idx_map]

    filtered_docs = [docs[i] for i in candidate_indices]
    filtered_matrix = matrix[candidate_indices]
    filtered_clusters = [clusters[i] for i in candidate_indices]
    filtered_filenames = [filenames[i] for i in candidate_indices]

    query_vector = vectorizer.transform([' '.join(expanded_words_query)])

    similarity_scores = []
    if method == "bm25":
        bm25_subset = BM25Okapi([doc.split() for doc in filtered_docs])
        scores = bm25_subset.get_scores(expanded_words_query)
        top_docs_indices = most_similar(scores, min_Doc=10)
        similarity_scores = [scores[i] for i in top_docs_indices]
    elif method == "hybrid":
        scores_tfidf = cos_similarity(query_vector, filtered_matrix)
        top_100_indices = most_similar(scores_tfidf, min_Doc=100)
        scores_bm25_full = BM25Okapi([doc.split() for doc in filtered_docs]).get_scores(expanded_words_query)
        scores_bm25_subset = np.array([scores_bm25_full[i] for i in top_100_indices])
        sorted_indices = np.argsort(scores_bm25_subset)[::-1]
        top_docs_indices = [top_100_indices[i] for i in sorted_indices[:10]]
        similarity_scores = [scores_bm25_subset[i] for i in sorted_indices[:10]]
    else:
        scores = cos_similarity(query_vector, filtered_matrix)
        top_docs_indices = most_similar(scores, min_Doc=10)
        similarity_scores = [scores[i] for i in top_docs_indices]

    spacy_model = spacy.load("en_core_web_sm")
    results = []
    for rank, idx in enumerate(top_docs_indices):
        doc_text = filtered_docs[idx]
        matches = list(datefinder.find_dates(doc_text))
        entity_doc = spacy_model(doc_text)
        entities = [(ent.text, ent.label_) for ent in entity_doc.ents]

        similarity_score = similarity_scores[rank] if rank < len(similarity_scores) else 0.0

        results.append({
            "filename": filtered_filenames[idx],
            "cluster": int(filtered_clusters[idx]),
            "dates_found": [str(d) for d in matches],
            "named_entities": entities,
            "similarity_score": float(similarity_score)
        })

    suggestions = suggest_queries_from_examples(' '.join(words_query), example_queries) if example_queries else []

    retrieved_doc_ids = []
    for idx in top_docs_indices:
        doc_id = filtered_filenames[idx].replace(".text", "")
        post_id = docid_to_postid.get(doc_id, doc_id)
        retrieved_doc_ids.append(str(post_id))

    if dataset == "wikIR1k":
        if WIKIR_QUERIES:
            wikir_queries_texts = [q["query"] for q in WIKIR_QUERIES]
            wikir_vectors = vectorizer.transform(wikir_queries_texts)
            query_similarities = cos_similarity(query_vector, wikir_vectors)
            best_idx = int(np.argmax(query_similarities))
            query_id = WIKIR_QUERIES[best_idx]["qid"]
        else:
            query_id = "unknown"
    else:
        if example_queries:
            query_similarities = [cos_similarity(vectorizer.transform([q]), query_vector)[0] for q in example_queries]
            best_idx = int(np.argmax(query_similarities))
            query_id = LIFESTYLE_QUERIES[best_idx]["qid"]
        else:
            query_id = "unknown"

    def evaluate_query(query_id, retrieved, relevance_data):
        relevant = relevance_data.get(query_id, [])
        def precision_at_k(relevant, retrieved, k=10):
            return len([d for d in retrieved[:k] if d in relevant]) / k if k > 0 else 0.0
        def recall(relevant, retrieved):
            return len([d for d in retrieved if d in relevant]) / len(relevant) if relevant else 0.0
        def average_precision(relevant, retrieved):
            hits, score = 0, 0.0
            for i, doc in enumerate(retrieved):
                if doc in relevant:
                    hits += 1
                    score += hits / (i + 1)
            return score / len(relevant) if relevant else 0.0
        def reciprocal_rank(relevant, retrieved):
            for i, doc in enumerate(retrieved):
                if doc in relevant:
                    return 1 / (i + 1)
            return 0.0
        return {
            "Precision@10": round(precision_at_k(relevant, retrieved), 4),
            "Recall": round(recall(relevant, retrieved), 4),
            "MAP": round(average_precision(relevant, retrieved), 4),
            "MRR": round(reciprocal_rank(relevant, retrieved), 4)
        }

    evaluation = evaluate_query(query_id, retrieved_doc_ids, relevance_data) if query_id in relevance_data else {
        "Precision@10": None,
        "Recall": None,
        "MAP": None,
        "MRR": None,
        "note": "لا يوجد تقييم لهذا الاستعلام"
    }

    expanded_set = set(expanded_words_query)
    top_docs = [filtered_docs[i] for i in top_docs_indices]
    top_docs_tokens = [set(doc.split()) for doc in top_docs]
    keyword_overlap = set()
    for doc_tokens in top_docs_tokens:
        keyword_overlap.update(expanded_set & doc_tokens)
    coverage_percent = len(keyword_overlap) / len(expanded_set) * 100 if expanded_set else 0.0

    if method in ["bm25", "hybrid"]:
        max_score = max(similarity_scores) if similarity_scores else 1
        normalized_scores = [(score / max_score) if max_score > 0 else 0 for score in similarity_scores]
        avg_similarity = np.mean(normalized_scores) * 100
    else:
        avg_similarity = np.mean(similarity_scores) * 100 if similarity_scores else 0.0

    return {
        "query_corrected": ' '.join(words_query),
        "query_expanded": ' '.join(expanded_words_query),
        "ranking_method": method,
        "top_similar_docs": results,
        "inverted_index_terms_count": len(inverted_index.index),
        "evaluation": evaluation,
        "evaluation_estimate": {
            "avg_similarity_percent@10": round(avg_similarity, 2),
            "keyword_coverage": round(coverage_percent, 2),
            "matched_keywords_total": len(keyword_overlap)
        },
        "suggested_queries": suggestions,
        "retrieved_docs_from_index": len(candidate_indices),
        "elapsed_time": round(time.time() - start_time, 4)
    }