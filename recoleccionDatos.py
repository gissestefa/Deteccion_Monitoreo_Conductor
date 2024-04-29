import mediapipe as mp
import cv2
from openpyxl import Workbook
import keyboard

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
isClosed = True
import math

book = Workbook()
sheet = book.active
fila = 1
cadena1 = 'A'
cadena2 = 'B'
cadena3 = 'C'
cadena4 = 'D'
cadena5 = 'E'

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
index_list = [34, 264, 175, 10]
with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        height, width, _ = image.shape
        image.flags.writeable = False
        results = face_mesh.process(image)
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        x1 = (0, 0)
        x2 = (0, 0)
        y1 = (0, 0)
        y2 = (0, 0)
        i = 0

        distancia1 = 0
        distancia2 = 0
        distancia3 = 0
        distancia4 = 0

        suma1 = 0
        suma2 = 0
        bandera = bool

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_list:

                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    if i == 0:
                        # primer punto eje x
                        x1 = (x, y)

                    if i == 1:
                        # segundo punto eje x
                        x2 = (x, y)
                        cv2.line(image, x1, x2, (255, 0, 255), 2)

                    if i == 2:
                        # primer punto eje y
                        y1 = (x, y)

                    if i == 3:
                        # segundo punto eje y
                        y2 = (x, y)
                        cv2.line(image, y1, y2, (255, 0, 255), 2)
                        i = 0
                        # linea entre ejes (-x,-y)
                        cv2.line(image, x1, y1, (255, 0, 255), 2)
                        # linea entre ejes (x,-y)
                        cv2.line(image, x2, y1, (255, 0, 255), 2)
                        # linea entre ejes (-x, y)
                        cv2.line(image, x1, y2, (255, 0, 255), 2)
                        # linea entre ejes (x, y)
                        cv2.line(image, x2, y2, (255, 0, 255), 2)
                        # Calculo de distancias
                        # Distancia entre ejes (-x,y)
                        #distancia1 = math.sqrt((y1[0] - x1[0]) ** 2 + (y1[1] - x1[1]) ** 2)
                        distancia1 = math.sqrt((y2[0] - x1[0]) ** 2 + (y2[1] - x1[1]) ** 2)
                        print("DISTANCIA 1")
                        print(distancia1)
                        # Distancia entre ejes (x,y)
                        #distancia2 = math.sqrt((y1[0] - x2[0]) ** 2 + (y1[1] - x2[1]) ** 2)
                        distancia2 = math.sqrt((y2[0] - x2[0]) ** 2 + (y2[1] - x2[1]) ** 2)
                        print("DISTANCIA 2")
                        print(distancia2)
                        # Distancia entre ejes (-x,-y)
                        #distancia3 = math.sqrt((y2[0] - x1[0]) ** 2 + (y2[1] - x1[1]) ** 2)
                        distancia3 = math.sqrt((y1[0] - x1[0]) ** 2 + (y1[1] - x1[1]) ** 2)
                        print("DISTANCIA 3")
                        print(distancia3)
                        # Distancia entre ejes (x,-y)
                        #distancia4 = math.sqrt((y2[0] - x2[0]) ** 2 + (y2[1] - x2[1]) ** 2)
                        distancia4 = math.sqrt((y1[0] - x2[0]) ** 2 + (y1[1] - x2[1]) ** 2)
                        print("DISTANCIA 4")
                        print(distancia4)

                        fila += 1
                        cadena1 = cadena1 + str(fila)
                        cadena2 = cadena2 + str(fila)
                        cadena3 = cadena3 + str(fila)
                        cadena4 = cadena4 + str(fila)
                        cadena5 = cadena5 + str(fila)
                        print(cadena1, cadena2, cadena3, cadena4)

                        if keyboard.read_key() == "a":
                            print("Giro Izquierda")
                            sheet[cadena1] = str(distancia1)
                            sheet[cadena2] = str(distancia2)
                            sheet[cadena3] = str(distancia3)
                            sheet[cadena4] = str(distancia4)
                            sheet[cadena5] = 'Izquierda'
                        elif keyboard.read_key() == "w":
                            print("Giro Arriba")
                            sheet[cadena1] = str(distancia1)
                            sheet[cadena2] = str(distancia2)

                            sheet[cadena3] = str(distancia3)
                            sheet[cadena4] = str(distancia4)
                            sheet[cadena5] = 'Arriba'
                        elif keyboard.read_key() == "d":
                            print("Giro Derecha")
                            sheet[cadena1] = str(distancia1)
                            sheet[cadena2] = str(distancia2)
                            sheet[cadena3] = str(distancia3)
                            sheet[cadena4] = str(distancia4)
                            sheet[cadena5] = 'Derecha'
                        elif keyboard.read_key() == "s":
                            print("Giro Abajo")
                            sheet[cadena1] = str(distancia1)
                            sheet[cadena2] = str(distancia2)
                            sheet[cadena3] = str(distancia3)
                            sheet[cadena4] = str(distancia4)
                            sheet[cadena5] = 'Abajo'
                        elif keyboard.read_key() == "q":
                            print("Posición Correcta")
                            sheet[cadena1] = str(distancia1)
                            sheet[cadena2] = str(distancia2)
                            sheet[cadena3] = str(distancia3)
                            sheet[cadena4] = str(distancia4)
                            sheet[cadena5] = 'Correcta'

                        cadena1 = 'A'
                        cadena2 = 'B'
                        cadena3 = 'C'
                        cadena4 = 'D'
                        cadena5 = 'E'

                    if i != 3:
                        i += 1

        cv2.imshow('MediaPipe FaceMesh', image)
        book.save('valores1000.xlsx')
        # book.read_only('valores_n.xlsx')
        # df = pd.read_excel('valores_n.xlsx', header=None)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()