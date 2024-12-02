# Contador De Piscadas
![Gif](https://github.com/AntonioABLima/Contador-De-Piscadas/blob/main/Media/MainGif.gif?raw=true)

Este é um projeto de um contador de piscadas usando análise de imagem implementado em Python. O objetivo é detectar o número de piscadas de olhos de uma pessoa a partir de uma imagem, vídeo ou qualquer tipo de mídia visual. Isso pode ser útil em diversas aplicações, como monitoramento de fadiga, análise de atenção e até mesmo para aprimorar experiências de realidade virtual, no meu caso utilizei para confirmar se o título do Orochinho de Pisca é merecido ou não.

## Requisitos
Antes de começar, certifique-se de ter o seguinte instalado em sua máquina:
- Python (versão 3.6 ou superior)
- Bibliotecas: OpenCV (cv2) e Media Pipe (mediapipe)

## Funcionamento
O funcionamento do contador de piscadas com análise de imagem utilizando Python, OpenCV e Mediapipe baseia-se na detecção precisa dos pontos faciais associados aos olhos em uma imagem de vídeo em tempo real. Através da combinação dessas bibliotecas, o sistema é capaz de identificar os contornos dos olhos e calcular a relação de aspecto dos mesmos. Essa relação é utilizada como um indicador para determinar se um piscar de olhos ocorreu. Quando a relação de aspecto dos olhos cai abaixo de um certo limiar, o sistema registra uma piscada e incrementa o contador correspondente. 

### Observação
O código foi reajustado para calibrar as piscadas indepente da resolução, ao iniciado o programa sugere que o usuário de algumas piscadas para a calibragem e a contagem é inicada.

### Vídeo do Projeto
*   [YouTube](https://youtu.be/iftimDe8hzA)


## Código:
```python
import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import math
import numpy as np

def _map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def main():
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FPS, 25.0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(height)
    ratios = []
    
    openedEyes = 500
    closedEyes = 0
    blinkMap = 0
    piscando = False
    blinkCount = 0
    calibrating = True 
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if(results.multi_face_landmarks):
                face_landmarks = results.multi_face_landmarks[0].landmark
                
            cord1 = _normalized_to_pixel_coordinates(
                face_landmarks[159].x, 
                face_landmarks[159].y, 
                width, 
                height)
            cord2 = _normalized_to_pixel_coordinates(
                face_landmarks[145].x, 
                face_landmarks[145].y, 
                width, 
                height)
            cord3 = _normalized_to_pixel_coordinates(
                face_landmarks[33].x, 
                face_landmarks[33].y, 
                width, 
                height)
            cord4 = _normalized_to_pixel_coordinates(
                face_landmarks[133].x, 
                face_landmarks[133].y, 
                width, 
                height)
            
            cv2.line(image, cord1, cord2, (255, 0, 0), 4)
            cv2.line(image, cord3, cord4, (0, 0, 255), 4)

            dist = math.sqrt((cord1[0] - cord2[0])**2 + (cord1[1] - cord2[1])**2)
            dist2 = math.sqrt((cord4[0] - cord3[0])**2 + (cord4[1] - cord3[1])**2)

            ratio = (dist2 / (dist+0.001))
            ratios.append(ratio)

            if len(ratios) == 5:
                ratios.pop(0)

            mediaRatio =  np.mean(ratios)
            ratioMapped = _map(mediaRatio, 0, 40, 40, 0)

            print(ratioMapped)
            
            if calibrating:

                if ratioMapped < openedEyes and piscando == False:
                    blinkCount += 1
                    openedEyes = ratioMapped

                    piscando = True

                if ratioMapped >= closedEyes and piscando == True:
                    closedEyes = ratioMapped
                    piscando = False

                if blinkCount >= 5:
                    blinkMap = closedEyes + (openedEyes - closedEyes) / 2
                    calibrating = False
                    blinkCount = 0
            else:
                print("---blink-map---",blinkMap)
                if ratioMapped < blinkMap and piscando == False:
                    blinkCount += 1
                    piscando = True

                if ratioMapped >= blinkMap + 1 and piscando == True:
                    piscando = False
            
            font = cv2.FONT_HERSHEY_SIMPLEX

            if calibrating:
                cv2.putText(image, "Pisque para calibrar o dispositivo", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, f"{blinkCount}", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Face Mesh', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
    cap.release()

if __name__ == '__main__':
    main()
```
