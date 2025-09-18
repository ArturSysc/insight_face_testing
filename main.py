import cv2
import numpy as np
import json
from insightface.app import FaceAnalysis
import os

# Inicializa modelo
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)  # CPU=-1 para usar CPU, 0 para GPU se disponível

DB_FILE = "faces_db.json"

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

def recognize_face(embedding, threshold=0.5):
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

# ------------------- Loop principal -------------------

cap = cv2.VideoCapture(0)  # webcam

print("Pressione 'c' para cadastrar rosto, 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        emb = face.embedding

        # Reconhecimento
        name, score = recognize_face(emb, threshold=0.5)

        if name:
            color = (0, 255, 0)
            text = f"{name} ({score:.2f})"
        else:
            color = (0, 0, 255)
            text = f"Desconhecido ({score:.2f})"

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Reconhecimento Facial", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c") and faces:
        nome = input("Digite o nome para cadastrar: ")
        add_face(nome, faces[0].embedding)

cap.release()
cv2.destroyAllWindows()
