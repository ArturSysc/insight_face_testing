import cv2

# Carregar o classificador Haar Cascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar captura de vídeo (0 para câmera padrão ou caminho do arquivo para vídeos)
cap = cv2.VideoCapture(0)

while True:
    # Captura frame a frame
    ret, frame = cap.read()
    
    if not ret:
        print("Falha na captura do vídeo")
        break

    # Converter a imagem para escala de cinza (necessário para a detecção de rostos)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhar um retângulo ao redor de cada rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Exibir o vídeo com os rostos detectados
    cv2.imshow('Video com Detecção de Rostos', frame)

    # Se pressionar 'q', encerra a captura
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
