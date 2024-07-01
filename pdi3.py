# -*- coding: utf-8 -*-
"""PDI3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NVQ_qIOQdNetfboGP8D78d0YG843G0C8
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Leer un video ------------------------------------------------
cap = cv2.VideoCapture('ruta_1.mp4')                # Abro el video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # Meta-Información del video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # No la usamos en este script,...
fps = int(cap.get(cv2.CAP_PROP_FPS))                # ... pero puede ser útil en otras ocasiones
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()

# Crear una máscara binaria del mismo tamaño que cada fotograma del video
mask = np.zeros((height, width), dtype=np.uint8)

puntos = np.array([[130,534],[905,534], [510, 316],[452,316]], dtype=np.int32)
while (cap.isOpened()):                                                 # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()                                             # Obtengo el frame
    if ret==True:                                                       # ret indica si la lectura fue exitosa (True) o no (False)
        # frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))  # Si el video es muy grande y al usar cv2.imshow() no entra en la pantalla, se lo puede escalar (solo para visualización!)

        #---- Proceso -----------------------------------------
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.fillPoly(mask, [puntos],255)
        # Aplicar la máscara a la imagen
        masked_image = cv2.bitwise_and(frame_HSV, frame_HSV, mask=mask)
        frame_threshold = cv2.inRange(masked_image, (0, 0, 171), (255, 255, 255))
        #---- Muestro ------------------------------------------
        #plt.imshow(frame_threshold, cmap='gray'), plt.show()
        edges = cv2.Canny(frame_threshold, 100, 170, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=5, maxLineGap=30)
        """for i in range(len(lines)):
            linea = lines[i][0]
            cv2.line(linea[0], etc..)
"""
        cv2.imshow('Frame',edges)                                       # Muestro el frame
        if cv2.waitKey(25) & 0xFF == ord('q'):                          # Corto la repoducción si se presiona la tecla "q"
            break
    else:
        break                                       # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.
cap.release()                   # Cierro el video
cv2.destroyAllWindows()         # Destruyo todas las ventanas abiertas