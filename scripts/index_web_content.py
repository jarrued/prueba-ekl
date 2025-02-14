import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
import time

# Esperar a que Elasticsearch esté disponible
time.sleep(30)

# URL de la página web de prueba
url = "https://es.wikipedia.org/wiki/Inteligencia_artificial"

# Obtener el contenido de la página web
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extraer el texto de los párrafos
paragraphs = soup.find_all('p')
texts = [p.get_text() for p in paragraphs]

# Inicializar el modelo de embeddings y el clasificador de sentimientos
model = SentenceTransformer('all-MiniLM-L6-v2')
classifier = pipeline('sentiment-analysis')

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