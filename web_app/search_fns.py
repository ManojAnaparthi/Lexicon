import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from tqdm import tqdm

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_doc_embedding(tokens, w2v_model):
    embeddings = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(w2v_model.vector_size)

def search(query, top_k, w2v_model, index, passage_metadata, dataset):
    query_tokens = word_tokenize(query.lower())
    query_embedding = get_doc_embedding(query_tokens, w2v_model).reshape(1, -1)
    query_embedding = query_embedding.astype(np.float32)
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        doc_id, passage_num = passage_metadata[idx]
        original_sample = dataset['train'][doc_id]
        results.append({
            'passage': original_sample['passages']['passage_text'][passage_num],
            'score': float(score),
            'is_selected': original_sample['passages']['is_selected'][passage_num],
            'query_id': original_sample['query_id']
        })
    return results

def hybrid_search(query, top_k, w2v_model, index, passage_metadata, dataset):
    initial_results = search(query, 100, w2v_model, index, passage_metadata, dataset)
    candidate_passages = [res['passage'] for res in initial_results]
    
    query_embedding = bert_model.encode([query])
    passage_embeddings = bert_model.encode(candidate_passages)
    scores = np.dot(passage_embeddings, query_embedding.T).flatten()
    
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [initial_results[i] for i in top_indices]

def bm25_rerank(query, top_k, w2v_model, index, passage_metadata, dataset):
    initial_results = search(query, 100, w2v_model, index, passage_metadata, dataset)
    tokenized_passages = [word_tokenize(p.lower()) for p in [res['passage'] for res in initial_results]]
    
    bm25 = BM25Okapi(tokenized_passages)
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [initial_results[i] for i in top_indices]