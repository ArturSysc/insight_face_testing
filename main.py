import cv2
import numpy as np
import json
import os
import time
from insightface.app import FaceAnalysis

# Inicializa modelo
app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=0)  # CPU=-1 para CPU, 0 para GPU

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
    best_match, best_score = None, -1
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

cap = cv2.VideoCapture(0)

print("Digite 'c' para cadastrar rosto, 'q' para sair, Enter para continuar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame.")
        break

    # Marca início da leitura
    start_time = time.time()

    faces = app.get(frame)

    # Marca fim da leitura
    elapsed = time.time() - start_time
    print(f"⏱ Tempo de leitura facial: {elapsed:.3f} segundos")

    for face in faces:
        emb = face.embedding
        name, score = recognize_face(emb, threshold=0.5)
        if name:
            print(f"😀 Reconhecido: {name} ({score:.2f})")
        else:
            print(f"❌ Desconhecido ({score:.2f})")

    # Pergunta para cadastro ou saída
    cmd = input("Comando [c=Cadastro, q=Sair, Enter=Continuar]: ").strip().lower()
    if cmd == "q":
        print("Encerrando...")
        break
    elif cmd == "c" and faces:
        nome = input("Digite o nome para cadastrar: ")
        add_face(nome, faces[0].embedding)

    # Delay de 2 segundos antes de capturar novamente
    time.sleep(2)

# Libera a câmera ao final
cap.release()
print("Câmera liberada. Programa encerrado.")
