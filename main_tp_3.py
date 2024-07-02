import cv2
import numpy as np
from funciones_tp_3 import calcular_linea_promedio, dibujar_linea_extrapolada

#Leer el video 
cap = cv2.VideoCapture('ruta_2.mp4')               
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
fps = int(cap.get(cv2.CAP_PROP_FPS))                
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()
#Crear una máscara binaria del mismo tamaño que cada fotograma del video
mask = np.zeros((height, width), dtype=np.uint8)

#Coordenadas del trapecio
puntos = np.array([[130,534],[905,534], [510, 316],[452,316]], dtype=np.int32)

while cap.isOpened():                                                 #Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()                                             #Obtengo el frame
    if ret==True:                                                       #ret indica si la lectura fue exitosa (True) o no (False)

        #Proceso
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.fillPoly(mask, [puntos], 255)

        #Aplicar la máscara a la imagen
        masked_image = cv2.bitwise_and(frame_HSV, frame_HSV, mask=mask)

        frame_threshold = cv2.inRange(masked_image, (0, 0, 171), (255, 255, 255))

        #Muestro
        edges = cv2.Canny(frame_threshold, 100, 170, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180*0.2, threshold=50, minLineLength=5, maxLineGap=200)

        if lines is not None:
            lineas_izquierda = []
            lineas_derecha = []
            for linea in lines:
                x1, y1, x2, y2 = linea[0]
                pendiente = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else np.inf
                if pendiente < 0:
                    lineas_izquierda.append(linea)
                else:
                    lineas_derecha.append(linea)
            
            # Calcular y dibujar la línea promedio para la izquierda
            pendiente_izquierda, intercepto_izquierda = calcular_linea_promedio(lineas_izquierda)
            if pendiente_izquierda is not None:
                y1, y2 = puntos[2][1], puntos[0][1]
                frame_copy = frame.copy()
                dibujar_linea_extrapolada(frame_copy, pendiente_izquierda, intercepto_izquierda, y1, y2, (255, 0, 0), 8)
                cv2.addWeighted(frame_copy, 0.5, frame, 0.5, 0, frame)
            
            #Calcular y dibujar la línea promedio para la derecha
            pendiente_derecha, intercepto_derecha = calcular_linea_promedio(lineas_derecha)
            if pendiente_derecha is not None:
                y1, y2 = puntos[3][1], puntos[1][1]
                frame_copy = frame.copy()
                dibujar_linea_extrapolada(frame_copy, pendiente_derecha, intercepto_derecha, y1, y2, (255, 0, 0), 8)
                cv2.addWeighted(frame_copy, 0.5, frame, 0.5, 0, frame)

        cv2.imshow('Frame',frame)                                       #Muestro el frame
        if cv2.waitKey(25) & 0xFF == ord('q'):                          #Corto la repoducción si se presiona la tecla "q"
            break
    else:
        break                                       #Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.
cap.release()                   #Cierro el video
cv2.destroyAllWindows()         #Destruyo todas las ventanas abiertas
