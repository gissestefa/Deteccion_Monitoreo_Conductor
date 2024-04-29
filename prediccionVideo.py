import mediapipe as mp
import cv2
import math
import pandas as pd
import openpyxl
from sklearn import tree
from Utils import convertirResultadoEntrenamiento
import imutils

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
isClosed = True

#Extraccion datos excel
ruta_archivo = "Datos/1000/entrenamiento1000.xlsx"
nombre_hoja_features = "features"
nombre_hoja_labels = "labels"

#Features
df1 = pd.read_excel(ruta_archivo, sheet_name = nombre_hoja_features,  header=None)
features=df1.values.tolist()

#Labels
doc = openpyxl.load_workbook (ruta_archivo)
hoja = doc.get_sheet_by_name(nombre_hoja_labels)
labels=[]
for row in hoja.iter_rows():
    res = row[0].value
    labels.append(res)

#Entrenamiento
classifier=tree.DecisionTreeClassifier()
classifier.fit(features, labels)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
index_list =[34,264,175,10]

cap = cv2.VideoCapture('Videos/manejo2_9.mp4')
#cap = cv2.VideoCapture('Videos/VideoEntrada5.mp4')

with mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,min_detection_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame= cv2.flip(frame,1)
        frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        frame = imutils.resize(frame,height=975,width=575)
        height, width, _ = frame.shape
        x1=(0,0)
        x2=(0,0)
        y1=(0,0)
        y2=(0,0)
        i=0
        distancia1=0
        distancia2=0
        distancia3=0
        distancia4=0

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
                        #cv2.line(image, x1, x2, (255, 0, 255), 2)

                    if i == 2:
                        #primer punto eje y
                        y1 = (x, y)

                    if i == 3:
                        #segundo punto eje y
                        y2 = (x, y)
                        #cv2.line(image, y1, y2, (255, 0, 255), 2)
                        i=0
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
                        distancia3 = math.sqrt((y1[0] - x1[0]) ** 2 + (y1[1] - x1[1]) ** 2)
                        print("DISTANCIA 3")
                        print(distancia3)
                        # Distancia entre ejes (x,-y)
                        distancia4 = math.sqrt((y1[0] - x2[0]) ** 2 + (y1[1] - x2[1]) ** 2)
                        print("DISTANCIA 4")
                        print(distancia4)
                        res=classifier.predict([[distancia1,distancia2,distancia3,distancia4]])
                        mensaje=convertirResultadoEntrenamiento(res)
                        cv2.putText(frame, mensaje, (80, 45), 2, 1, (255, 0, 255), 2, cv2.LINE_4)

                    if i != 3:
                        i+=1

        cv2.imshow("PRUEBA AUTO", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()