from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import requests
import json

app = Flask(__name__)

# Inicializar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query_text = request.form['query']
    query_embedding = model.encode([query_text])[0]

    query = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {
                        "query_vector": query_embedding.tolist()
                    }
                }
            }
        }
    }

    response = requests.get('http://elasticsearch:9200/my_index/_search', headers={"Content-Type": "application/json"}, data=json.dumps(query))
    results = response.json()
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)