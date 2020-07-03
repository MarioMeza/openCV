#importar las bibliotecas necesarias
import cv2
import imutils 

#indicamos los archivos de entrada haar, video
archivo_haar = '/home/miguel/Desktop/practicas/autos/autos.xml'
video_entrada = '/home/miguel/Desktop/practicas/autos/video2.avi'

video_salida = cv2.VideoWriter('video_salida.avi', cv2.VideoWriter_fourcc('D','I','V','X'), 30, (320,240), True)

#inicializar la lectura del video y archivo haar
cap = cv2.VideoCapture(video_entrada)
auto_detector = cv2.CascadeClassifier(archivo_haar)

#se procesan cada una de las imagenes (frames) contenido en el video
while True:
	ret, img = cap.read()
	autos = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#deteccion de autos
	autos = auto_detector.detectMultiScale(autos, 1.1, 1)

	#extraccion de las coordenadas de los autos encontrados
	for (x,y,w,h) in autos:
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

	#escribimos cada uno de los frames procesados	
	video_salida.write(img)

	#mostramos la imagen procesada
	cv2.imshow('video', img)

	#interrupcion del proceso
	if cv2.waitKey(33) == 27:
		video_salida.release()
		break

video_salida.release()
cv2.destroyAllWindows()
