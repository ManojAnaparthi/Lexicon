# 🔍 Lexicon: Word Embedding Based Search Engine

**Lexicon** is a semantic search engine built using modern word embeddings and hybrid retrieval methods. Built on a 100,000-passage subset of the **MS MARCO** dataset, this project evaluates and compares multiple retrieval pipelines:

- **Word2Vec only**
- **Word2Vec + BERT re-ranking**
- **Word2Vec + BM25 re-ranking**

## 📊 Results at a Glance

| Retrieval Setup       | MRR@10       | Recall@10   | NDCG@10     |
|-----------------------|--------------|-------------|-------------|
| Word2Vec              | 0.1568       | 0.450       | 0.2244      |
| Word2Vec + **BERT**   | **0.1983**   | **0.525**   | **0.2728**  |
| Word2Vec + BM25       | 0.1701       | 0.478       | 0.2404      |

## 🎥 Live Demo

👉 [Streamlit Demo Video]([https://iitgnacin-my.sharepoint.com/:v:/g/personal/23110071_iitgn_ac_in/Eetee1CE1HdKpPXwFzSvb84BzeRomRpsBKF4EnEJVmJzLA?e=XAthoS](https://iitgnacin-my.sharepoint.com/:v:/g/personal/23110025_iitgn_ac_in/EResIprMMyJJhK52XxIrGIMBf4Hs5K4U9RNmkhnDAvfvZw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=ANJs5f))

## ⚙️ Features

- 🔹 Fast semantic retrieval with **Word2Vec** and **FAISS**
- 🔹 Enhanced re-ranking using **Sentence-BERT (all-MiniLM-L6-v2)**
- 🔹 Traditional lexical matching with **BM25**
- 🔹 Clustering and visualization with **t-SNE** and **K-Means**

## 🛠 Methodology

### 1. Word2Vec Baseline
- 300-dim embeddings trained with Gensim
- Passage representation = mean of word vectors
- Approximate nearest neighbors via FAISS

### 2. Word2Vec + BERT
- Initial candidates from FAISS + Word2Vec
- Re-ranked using cosine similarity of Sentence-BERT embeddings

### 3. Word2Vec + BM25
- Combined ranking using normalized FAISS scores and BM25 scores

## 📁 Repository Structure

Lexicon/<br>
├── data/            # Processed MS MARCO subset and utility data files (**This has been done locally and the dataset size was too large to push to Github**)<br>
├── model/           # Saved Word2Vec and BERT models<br>
├── notebook/        # Jupyter notebooks for development and analysis<br>
├── web_app/        # Streamlit app code for demo<br>
├── Project_Report_Lexicon.pdf  # Final report<br>
├── requirements.txt # Python dependencies<br>
└── README.md       # You're here


## 📚 Tools & Libraries

- [MS MARCO](https://huggingface.co/datasets/msmarco)
- [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) - Word2Vec
- [FAISS](https://faiss.ai/) - Fast similarity search
- [NLTK](https://www.nltk.org/) - Text preprocessing
- [SentenceTransformers](https://www.sbert.net/) - BERT embeddings
- [Scikit-learn](https://scikit-learn.org/) - Clustering
- [Matplotlib](https://matplotlib.org/) - Visualization

## 📈 Evaluation Metrics

- **MRR@10** - Mean Reciprocal Rank
- **Recall@10**
- **NDCG@10**
- **Semantic Similarity (Mean & Max)**
- **Hit Rate@5**

## 👥 Contributors

- **A.V.S Manoj** (23110025) – [manoj.anaparthi@iitgn.ac.in](mailto:manoj.anaparthi@iitgn.ac.in)  
- **B. Saharsh** (23110071) – [burra.saharsh@iitgn.ac.in](mailto:burra.saharsh@iitgn.ac.in)  
- **O. Akash** (23110225) – [23110225@iitgn.ac.in](mailto:23110225@iitgn.ac.in)

---
