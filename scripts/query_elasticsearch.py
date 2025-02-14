import requests
import json
from sentence_transformers import SentenceTransformer
import time

# Esperar a que Elasticsearch esté disponible
time.sleep(30)

# Inicializar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Texto de la consulta
query_text = "aplicaciones de inteligencia artificial"
query_embedding = model.encode([query_text])[0]

# Crear la consulta para Elasticsearch
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

# Realizar la consulta en Elasticsearch
response = requests.get('http://elasticsearch:9200/my_index/_search', headers={"Content-Type": "application/json"}, data=json.dumps(query))
print(response.json())