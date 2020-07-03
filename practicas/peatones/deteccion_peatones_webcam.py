#importar las bibliotecas necesarias
import numpy as np
import imutils
import cv2
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt

# se inicializa descriptor HOG y SVM
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#inicializamos la captura de imagenes desde webcam
cap = cv2.VideoCapture(0)


#ciclo para que siempre este leyendo
while True:

	#captura de images
	ret, imagen = cap.read()
	imagen = cv2.flip(imagen, 1) #-1
	imagen = imutils.resize(imagen, width = 450)
	#detectar peatones en la imagen
	(rectas, weights) = hog.detectMultiScale(imagen, winStride= (4,4), padding =(8,8), scale = 1.05)
	#enmarcar peatones detectados
	rectas = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rectas])
	#suprimir multiples selecciones para un peaton
	eleccion = non_max_suppression(rectas, probs = None, overlapThresh = 0.65)
	#dibujar cuadros finales en imagen
	for (xA, yA, xB, yB) in eleccion:
		cv2.rectangle(imagen, (xA, yA), (xB, yB), (0, 255, 0), 2)
	#mostrar numero de peatones encontrados
	if (len(eleccion)):
		print("{} peatones encontrados".format(len(eleccion)))

	cv2.imshow("Imagen de salida", imagen)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
