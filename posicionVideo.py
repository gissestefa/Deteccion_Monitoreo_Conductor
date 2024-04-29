import mediapipe as mp
import cv2
import imutils
import numpy as np
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
isClosed = True
import math
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Videos/manejo2.mp4')
index_list =[34,264,175,10]
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, image = cap.read()
        if ret == False:
            break
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        image.flags.writeable = False
        results = face_mesh.process(image)
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        x1=(0,0)
        x2=(0,0)
        y1=(0,0)
        y2=(0,0)
        i=0
        distancia1=0
        distancia2=0
        distancia3=0
        distancia4=0
        sensibilidad=1.65
        sensibilidad2=1.50
        sensibilidad3=1.15
        suma1=0
        suma2=0
        image = imutils.resize(image, width=400)

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_list:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    if i == 0:
                        #primer punto eje x
                        x1= (x,y)

                    if i == 1:
                        #segundo punto eje x
                        x2 = (x, y)
                        cv2.line(image, x1, x2, (255, 0, 255), 2)

                    if i == 2:
                        #primer punto eje y
                        y1 = (x, y)

                    if i == 3:
                        #segundo punto eje y
                        y2 = (x, y)
                        cv2.line(image, y1, y2, (255, 0, 255), 2)
                        i=0
                        #linea entre ejes (-x,-y)
                        cv2.line(image, x1, y1, (255, 0, 255), 2)
                        # linea entre ejes (x,-y)
                        cv2.line(image, x2, y1, (255, 0, 255), 2)
                        # linea entre ejes (-x, y)
                        cv2.line(image, x1, y2, (255, 0, 255), 2)
                        # linea entre ejes (x, y)
                        cv2.line(image, x2, y2, (255, 0, 255), 2)
                        #Calculo de distancias
                        #Distancia entre ejes (-x,y)
                        distancia1 = math.sqrt((y2[0] - x1[0]) ** 2 + (y2[1] - x1[1]) ** 2)
                        print("DISTANCIA 1")
                        print(distancia1)
                        # Distancia entre ejes (x,y)
                        distancia2 = math.sqrt((y2[0] - x2[0]) ** 2 + (y2[1] - x2[1]) ** 2)
                        print("DISTANCIA 2")
                        print(distancia2)
                        # Distancia entre ejes (-x,-y)
                        distancia3 = math.sqrt((y1[0] - x2[0]) ** 2 + (y1[1] - x2[1]) ** 2)
                        print("DISTANCIA 3")
                        print(distancia3)
                        # Distancia entre ejes (x,-y)
                        distancia4 = math.sqrt((y1[0] - x1[0]) ** 2 + (y1[1] - x1[1]) ** 2)
                        print("DISTANCIA 4")
                        print(distancia4)
                        suma1 = distancia1+distancia2
                        suma2 = distancia3+distancia4

                        if distancia1 > (1.45*distancia2):
                            cv2.putText(image, 'Giro de cabeza derecha', (10, 45), 1, 1, (0, 255, 255), 2, cv2.LINE_4)
                            break

                        if distancia2 > (1.40*distancia1):
                            cv2.putText(image, 'Giro de cabeza izquierda', (10, 45), 1, 1, (0, 255, 255), 2, cv2.LINE_4)
                            break

                        if suma1 >= 1.5 * (suma2 / 2):
                            cv2.putText(image, 'Giro de cabeza arriba', (200, 45), 1, 1, (0, 255, 255), 2, cv2.LINE_4)
                            break

                        if suma2 >= (1.5*suma1):
                            cv2.putText(image, 'Giro de cabeza abajo', (200, 45), 1, 1, (0, 255, 255), 2, cv2.LINE_4)
                            break

                        cv2.putText(image, 'Posicion Correcta', (10, 45), 1, 1, (0, 255, 255), 2, cv2.LINE_4)

                    if i != 3:
                        i += 1

        cv2.imshow('DETECCION', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()