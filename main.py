import cv2
import numpy as np
import json
import os
from insightface.app import FaceAnalysis
import requests
import time

# ------------------- Configurações -------------------

DB_FILE = "faces_db.json"
API_URL = "http://SEU_ENDPOINT/api"  # substitua pelo endpoint da sua API
THRESHOLD = 0.5  # limiar de similaridade para reconhecer rosto

# ------------------- Inicializa modelo -------------------

app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=0)  # GPU=0, CPU=-1

# ------------------- Funções auxiliares -------------------

def load_db():
    """Carrega o banco de embeddings do JSON"""
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(db):
    """Salva o banco de embeddings no JSON"""
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

def add_face(name, embedding):
    """Adiciona um novo rosto ao banco"""
    db = load_db()
    db.append({"name": name, "embedding": embedding.tolist()})
    save_db(db)
    print(f"✅ Rosto de {name} cadastrado!")

def compare_embeddings(emb1, emb2):
    """Compara dois embeddings e retorna similaridade (cosine similarity)"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def recognize_face(embedding, threshold=THRESHOLD):
    """Compara embedding com todos cadastrados"""
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
        return best_match, best_score
    return None, best_score

def post_to_api(name, score):
    """Envia os dados para a API"""
    try:
        data = {"name": name, "score": float(score), "timestamp": int(time.time())}
        response = requests.post(API_URL, json=data, timeout=2)
        if response.status_code == 200:
            print("✅ API notificada com sucesso")
        else:
            print(f"⚠️ Erro na API: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Falha ao enviar para API: {e}")

# ------------------- Loop principal -------------------

cap = cv2.VideoCapture(0)
print("Pressione 'c' para cadastrar rosto, 'q' para sair.")

# Para evitar múltiplos POSTs para o mesmo rosto em sequência
last_detected = {}  # {"nome": timestamp}

POST_COOLDOWN = 5  # segundos entre envios para o mesmo rosto

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da câmera.")
        break

    faces = app.get(frame)
    for face in faces:
        emb = face.embedding

        # Reconhecimento
        name, score = recognize_face(emb)

        if name:
            print(f"😀 Reconhecido: {name} ({score:.2f})")
        else:
            print(f"❌ Rosto desconhecido ({score:.2f})")
            name = "desconhecido"

        # --- POST para a API com cooldown ---
        now = time.time()
        if name not in last_detected or (now - last_detected[name] > POST_COOLDOWN):
            post_to_api(name, score)
            last_detected[name] = now

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Encerrando...")
        break
    elif key == ord("c") and faces:
        nome = input("Digite o nome para cadastrar: ")
        add_face(nome, faces[0].embedding)

cap.release()
cv2.destroyAllWindows()
