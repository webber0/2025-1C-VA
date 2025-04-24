import cv2 as cv
import numpy as np

def cargar_contornos_referencia():
    nombres = ['circulo', 'cuadrado', 'triangulo']
    archivos = ['./formas/circulo4.png', './formas/cuadrado4.png', './formas/triangulo4.png']
    contornos = []

    for archivo in archivos:
        img = cv.imread(archivo, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"No se pudo cargar: {archivo}")
            continue

        _, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        contorno, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contorno:
            contorno_mayor = max(contorno, key=cv.contourArea)
            contornos.append(contorno_mayor)
        else:
            print(f"No se encontró contorno en: {archivo}")

        # Para test
        # cv.imshow(f"Original - {archivo}", img)
        # cv.imshow(f"Umbralizada - {archivo}", thresh)
        # cv.waitKey(0)

    return nombres, contornos

webcam = cv.VideoCapture(0)

# Barra para el umbral de binarización
cv.namedWindow("Ajustes")
cv.createTrackbar("Umbral binarizacion", "Ajustes", 100, 255, lambda x: None)
cv.createTrackbar("Tamaño estructura", "Ajustes", 1, 20, lambda x: None)
cv.createTrackbar("Umbral matchShapes", "Ajustes", 20, 100, lambda x: None)

nombres_ref, contornos_ref = cargar_contornos_referencia()

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Región de Interés (ROI)
    roi = frame.copy()

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # Obtenemos el umbral desde la barra
    thresh_val = cv.getTrackbarPos("Umbral binarizacion", "Ajustes")
    _, binary = cv.threshold(gray, thresh_val, 255, cv.THRESH_BINARY_INV)

    # Acá lo mismo pero con el tamaño de la estructura para operaciones morfológicas
    tam_estruc = cv.getTrackbarPos("Tamaño estructura", "Ajustes")
    if tam_estruc < 1:
        tam_estruc = 1
    kernel = np.ones((tam_estruc, tam_estruc), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    # Detecto los contornos
    contornos, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        area = cv.contourArea(cnt)
        if area < 1000:  # Se filtran los contornos cuya area es muy pequeña
            continue

        x, y, w, h = cv.boundingRect(cnt)
        match_val_min = float('inf')
        nombre_clase = "Desconocido"

        umbral_match = cv.getTrackbarPos("Umbral matchShapes", "Ajustes") / 100.0

        for i, cnt_ref in enumerate(contornos_ref):
            match_val = cv.matchShapes(cnt, cnt_ref, 1, 0.0)
            if match_val < umbral_match and match_val < match_val_min:
                match_val_min = match_val
                nombre_clase = nombres_ref[i]

        # Dibujo el contorno
        color = (0, 255, 0) if nombre_clase != "Desconocido" else (0, 0, 255)
        cv.drawContours(roi, [cnt], -1, color, 2)
        cv.putText(roi, nombre_clase, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # De esta manera mostramos 2 ventanas por separado
    # cv.imshow("Original", roi)
    # cv.imshow("Binaria", binary)

    # Acá juntamos las 2 ventanas y cambiamos el tamaño para ajustarlo mejor a la pantalla 
    bin_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)  # Convertir a color para concatenar
    combinada = cv.hconcat([roi, bin_color])
    escala = 0.5
    combinada = cv.resize(combinada, (int(combinada.shape[1] * escala), int(combinada.shape[0] * escala)))

    cv.imshow("Vista combinada", combinada)

    if cv.waitKey(1) & 0xFF == 27: # Tecla 'Esc' para salir
        break

webcam.release()
cv.destroyAllWindows()