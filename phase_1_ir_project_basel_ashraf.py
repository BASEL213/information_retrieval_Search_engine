# phase_1_ir_project_basel_ashraf.py
import pandas as pd
import pyterrier as pt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import re
import os
import zipfile
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import BadRequest
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize NLTK resources (download only if not already present)
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[\.\,\#_\|\:\?\?\/\=\@]", " ", text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    word_tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    processed_text = [stemmer.stem(word) for word in word_tokens if word not in stop_words]
    return " ".join(processed_text)

# Load and preprocess CISI dataset
def load_cisi_dataset(data_dir):
    def read_documents(documents_path):
        with open(documents_path, 'r') as file:
            lines = file.readlines()
        documents = []
        current_document = None
        for line in lines:
            if line.startswith('.I'):
                if current_document is not None:
                    current_document['Text'] = current_document['Text'].split('\t')[0].strip()
                    documents.append(current_document)
                current_document = {'ID': line.strip().split()[1], 'Text': ''}
            elif line.startswith('.T'):
                continue
            elif line.startswith('.A') or line.startswith('.B') or line.startswith('.W') or line.startswith('.X'):
                continue
            else:
                current_document['Text'] += line.strip() + ' '
        if current_document is not None:
            current_document['Text'] = current_document['Text'].split('\t')[0].strip()
            documents.append(current_document)
        return pd.DataFrame(documents)

    def read_queries(queries_path):
        with open(queries_path, 'r') as file:
            lines = file.readlines()
        query_texts = []
        query_ids = []
        current_query_id = None
        current_query_text = []
        for line in lines:
            if line.startswith('.I'):
                if current_query_id is not None:
                    query_texts.append(' '.join(current_query_text))
                    current_query_text = []
                current_query_id = line.strip().split()[1]
                query_ids.append(current_query_id)
            elif line.startswith('.W'):
                continue
            elif line.startswith('.X'):
                break
            else:
                current_query_text.append(line.strip())
        query_texts.append(' '.join(current_query_text))
        return pd.DataFrame({'qid': query_ids, 'raw_query': query_texts})

    def read_qrels(qrels_path):
        return pd.read_csv(qrels_path, sep='\s+', names=['qid', 'Q0', 'docno', 'label'])

    documents_path = os.path.join(data_dir, 'CISI.ALL')
    queries_path = os.path.join(data_dir, 'CISI.QRY')
    qrels_path = os.path.join(data_dir, 'CISI.REL')
    documents_df = read_documents(documents_path)
    queries_df = read_queries(queries_path)
    qrels_df = read_qrels(qrels_path)
    return documents_df, queries_df, qrels_df

# Extract and load dataset
zip_file_name = 'cisi.zip'
if not os.path.exists('cisi_dataset'):
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall('cisi_dataset')

data_dir = 'cisi_dataset'
documents_df, queries_df, qrels_df = load_cisi_dataset(data_dir)

# Prepare data
documents_df["docno"] = documents_df["ID"].astype(str)
documents_df['processed_text'] = documents_df['Text'].apply(preprocess)
queries_df["qid"] = queries_df["qid"].astype(str)
queries_df["query"] = queries_df["raw_query"].apply(preprocess)

# Initialize PyTerrier
if not pt.java.started():
    pt.java.init()

# Create index using absolute path
index_dir = os.path.abspath("myFirstIndex")
os.makedirs(index_dir, exist_ok=True)  # Ensure directory exists
indexer = pt.terrier.IterDictIndexer(index_dir, overwrite=True)
index_ref = indexer.index(documents_df[["processed_text", "docno"]].rename(columns={"processed_text": "text"}).to_dict(orient='records'))
index = pt.IndexFactory.of(index_ref)

# Retrieval models
tfidf_retr = pt.BatchRetrieve(index, controls={"wmodel": "TF_IDF"}, num_results=10)
bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=10)
rm3_expander = pt.rewrite.RM3(index, fb_terms=10, fb_docs=100)
rm3_qe = bm25 >> rm3_expander

# Initialize Sentence-BERT
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
# Precompute document embeddings
document_embeddings = sbert_model.encode(documents_df['Text'].tolist(), batch_size=32, show_progress_bar=True)
documents_df['embedding'] = list(document_embeddings)

# Synonym expansion (for non-SBERT models)
def expand_query_with_synonyms(query):
    synonyms = []
    for word in query.split():
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonyms.append(lemma.name())
    expanded_query = query + " " + " ".join(set(synonyms))
    return expanded_query

# Flask app
app = Flask(__name__, static_folder='src', static_url_path='/static', template_folder='templates')

@app.route("/", methods=['GET'])
def get_index():
    return render_template("index.html")

@app.route("/search", methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query')
        model = data.get('model', 'bm25')  # Default to BM25
        if not query:
            return jsonify({"error": "Empty query"}), 400

        # Preprocess query for non-SBERT models
        processed_query = preprocess(query)

        # Select retrieval model
        if model == "tfidf":
            results = tfidf_retr.search(processed_query)
        elif model == "bm25":
            results = bm25.search(processed_query)
        elif model == "sbert":
            # SBERT semantic search
            query_embedding = sbert_model.encode([query])[0]
            similarities = cosine_similarity([query_embedding], document_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:10]  # Top 10 results
            results = pd.DataFrame({
                'docno': documents_df.iloc[top_indices]['docno'],
                'score': similarities[top_indices]
            })
        else:
            return jsonify({"error": "Invalid model"}), 400

        # Format results
        result_list = []
        for _, row in results.iterrows():
            doc = documents_df[documents_df['docno'] == row['docno']].iloc[0]
            result_list.append({
                "title": f"Document {row['docno']}",
                "url": f"#doc{row['docno']}",  # Placeholder URL
                "description": doc['Text'][:150] + "..." if len(doc['Text']) > 150 else doc['Text']
            })

        # Query expansion for suggestions (skip for SBERT)
        suggestions = []
        if model == "rm3":
            expanded_query = rm3_qe.search(processed_query).iloc[0]["query"]
            suggestions = expanded_query.split()[1:]
        elif model != "sbert":
            suggestions = expand_query_with_synonyms(processed_query).split()

        return jsonify({
            "results": result_list,
            "suggestions": list(set(suggestions))[:5]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)