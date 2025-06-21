import streamlit as st
from search_fns import search, hybrid_search, bm25_rerank
import numpy as np
import faiss
from gensim.models import Word2Vec
from datasets import load_dataset

# Load dataset and models
def load_data_and_models():
    dataset = load_dataset("ms_marco", "v2.1", split={
        'train': 'train[:100000]',
        'validation': 'validation[:100000]',
        'test': 'test[:100000]'
    })
    
    dataset = dataset.remove_columns(['answers', 'wellFormedAnswers', 'query_type'])

    w2v_model = Word2Vec.load("word2vec_model.model")

    doc_embeddings = np.load("doc_embeddings.npy")
    doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
    faiss.normalize_L2(doc_embeddings)
    index = faiss.IndexFlatIP(300)
    index.add(doc_embeddings.astype('float32'))

    passage_metadata = []
    for doc_id, sample in enumerate(dataset['train']):
        for passage_num in range(len(sample['passages']['passage_text'])):
            passage_metadata.append((doc_id, passage_num))
    
    return dataset, w2v_model, index, passage_metadata

def main():
    st.title("🔎 LEXICON - Search Engine")
    st.write("Powered by Word2Vec, BERT & BM25")

    dataset, w2v_model, index, passage_metadata = load_data_and_models()

    query = st.text_input("Enter your search query:")

    model_choice = st.selectbox(
        "Select retrieval model:",
        ["Word2Vec", "Word2Vec + BERT", "Word2Vec + BM25"]
    )

    top_k = st.slider("Number of results:", min_value=1, max_value=20, value=10)

    snippet_len = 100

    if st.button("Search") and query:
        st.write(f"### Results using {model_choice}")

        if model_choice == "Word2Vec":
            results = search(query, top_k, w2v_model, index, passage_metadata, dataset)
        elif model_choice == "Word2Vec + BERT":
            results = hybrid_search(query, top_k, w2v_model, index, passage_metadata, dataset)
        elif model_choice == "Word2Vec + BM25":
            results = bm25_rerank(query, top_k, w2v_model, index, passage_metadata, dataset)

        seen_snippets = set()
        cleaned_results = []
        for res in results:
            passage = res['passage'].strip()
            snippet = passage[:snippet_len]
            if passage and snippet not in seen_snippets:
                seen_snippets.add(snippet)
                cleaned_results.append(res)

        for i, res in enumerate(cleaned_results, 1):
            st.markdown(f"> {res['passage']}")
            st.markdown("---")

if __name__ == "__main__":
    main()