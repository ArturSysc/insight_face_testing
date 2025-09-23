import cv2
import numpy as np
import json
import os
import time
from insightface.app import FaceAnalysis
import requests

# ------------------- Configurações -------------------

DB_FILE = "faces_db.json"
API_URL_RECOGNITION = "http://192.168.0.29:8000/recognitions"  # endpoint para enviar nomes
THRESHOLD = 0.5  # limiar de similaridade para reconhecer rosto

SEND_INTERVAL = 5  # segundos entre envios por pessoa
last_sent = {}     # dicionário: { "nome": timestamp }

# ------------------- Inicializa modelo -------------------

app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=0, det_size=(640, 640))

# ------------------- Funções auxiliares -------------------

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

def add_face(name, embedding):
    db = load_db()
    db.append({"name": name, "embedding": embedding.tolist()})
    save_db(db)
    print(f"✅ Rosto de {name} cadastrado!")

def compare_embeddings(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def recognize_face(embedding, threshold=THRESHOLD):
    db = load_db()
    best_match = None
    best_score = -1
    for person in db:
        emb_db = np.array(person["embedding"])
        score = compare_embeddings(embedding, emb_db)
        if score > best_score:
            best_score = score
            best_match = person["name"]
    if best_score >= threshold:
        return best_match
    return None

def post_name_only(name):
    """Envia apenas o nome para a API"""
    try:
        data = {"name": name, "timestamp": int(time.time())}
        response = requests.post(API_URL_RECOGNITION, json=data, timeout=2)
        if response.status_code == 200:
            print(f"✅ Nome enviado: {name}")
        else:
            print(f"⚠️ Erro na API: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Falha ao enviar nome: {e}")

# ------------------- Loop principal -------------------

cap = cv2.VideoCapture(0)
print("Rodando... pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da câmera.")
        break

    faces = app.get(frame)
    now = time.time()

    for face in faces:
        emb = face.embedding
        if emb is None:
            continue

        # Reconhecimento apenas do nome
        name = recognize_face(emb)
        if not name:
            name = "desconhecido"

        # --- Limitador de envio ---
        if name not in last_sent or now - last_sent[name] >= SEND_INTERVAL:
            post_name_only(name)
            last_sent[name] = now
        else:
            restante = SEND_INTERVAL - (now - last_sent[name])
            print(f"⏸️ Ignorando envio (aguarde {restante:.1f}s para {name})")

    # --- Tecla de saída ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Encerrando...")
        break

cap.release()
cv2.destroyAllWindows()
