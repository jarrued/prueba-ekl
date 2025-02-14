from sentence_transformers import SentenceTransformer
from transformers import pipeline
import requests
import json
import time

# Esperar a que Elasticsearch esté disponible
time.sleep(30)

# Inicializar el modelo de embeddings y el clasificador de sentimientos
model = SentenceTransformer('all-MiniLM-L6-v2')
classifier = pipeline('sentiment-analysis')

# Textos que deseas indexar
texts = ["Este es un ejemplo de texto.", "Otro ejemplo de texto."]

# Generar embeddings para los textos
embeddings = model.encode(texts)

# Indexar los documentos en Elasticsearch
for i, text in enumerate(texts):
    sentiment = classifier(text)[0]
    doc = {
        "text": text,
        "embedding": embeddings[i].tolist(),
        "sentiment": sentiment
    }
    response = requests.post('http://elasticsearch:9200/my_index/_doc', headers={"Content-Type": "application/json"}, data=json.dumps(doc))
    print(response.json())