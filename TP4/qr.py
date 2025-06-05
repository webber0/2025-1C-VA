import cv2
import numpy as np

# Variables globales
homography_matrix = None
grid_size = (300, 300)
clicked_points = []
collecting_points = False
display_message = "" # Variable para almacenar el mensaje a mostrar
color_code = ''

# Inicializar detector de QR
qr_detector = cv2.QRCodeDetector()

# Mouse callback para puntos manuales
def mouse_callback(event, x, y, flags, param):
    global clicked_points, collecting_points, display_message
    if event == cv2.EVENT_LBUTTONDOWN and collecting_points:
        clicked_points.append((x, y))
        if len(clicked_points) == 4:
            collecting_points = False
            display_message = "Homografía a partir de puntos manuales calculada."
            color_code = 'G'

# Dibuja grilla en perspectiva sobre la imagen original
def draw_grid(image, H, grid=(3, 3), grid_size=300):
    H_inv = np.linalg.inv(H)
    step_x = grid_size / grid[0]
    step_y = grid_size / grid[1]

    for i in range(1, grid[0]):
        x = i * step_x
        pt1 = np.float32([[x, 0]]).reshape(-1, 1, 2)
        pt2 = np.float32([[x, grid_size]]).reshape(-1, 1, 2)
        dst1 = cv2.perspectiveTransform(pt1, H_inv)
        dst2 = cv2.perspectiveTransform(pt2, H_inv)
        cv2.line(image,
                    tuple(np.int32(dst1[0][0])),
                    tuple(np.int32(dst2[0][0])),
                    (0, 255, 0), 1)

    for j in range(1, grid[1]):
        y = j * step_y
        pt1 = np.float32([[0, y]]).reshape(-1, 1, 2)
        pt2 = np.float32([[grid_size, y]]).reshape(-1, 1, 2)
        dst1 = cv2.perspectiveTransform(pt1, H_inv)
        dst2 = cv2.perspectiveTransform(pt2, H_inv)
        cv2.line(image,
                 tuple(np.int32(dst1[0][0])),
                 tuple(np.int32(dst2[0][0])),
                 (0, 255, 0), 1)
            
    return image

# Inicializar cámara
cap = cv2.VideoCapture(0)
cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", mouse_callback)

display_message = "Presioná 'q' para detectar QR, 'h' para clic manual. ESC para salir."
color_code = 'G'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Mostrar grilla si hay homografía válida
    if homography_matrix is not None:
        display = draw_grid(display, homography_matrix)
        frontal_view = cv2.warpPerspective(frame, homography_matrix, grid_size)
        cv2.imshow("Vista Frontal", frontal_view)

    # Mostrar puntos seleccionados con mouse
    for pt in clicked_points:
        cv2.circle(display, pt, 5, (0, 0, 255), -1)

    # Mostrar mensajes en la pantalla
    if color_code == 'G':
        COLOR = (0,255,0)
    elif color_code == 'B':
        COLOR = (0,0,255)
    elif color_code == 'R':
        COLOR = (255,0,0)
    elif color_code == 'Y':
        COLOR = (255,255,0)

    cv2.putText(display, display_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2, cv2.LINE_AA)


    cv2.imshow("Webcam", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        display_message = "Buscando código QR..."
        color_code = 'Y'
        retval, points = qr_detector.detect(frame)
        if retval and points is not None and len(points) == 1:
            pts = points[0].astype(np.float32)
            dst_pts = np.float32([[0, 0], [grid_size[0], 0], [grid_size[0], grid_size[1]], [0, grid_size[1]]])
            homography_matrix, _ = cv2.findHomography(pts, dst_pts)
            display_message = "Homografía a partir de QR calculada."
            color_code = 'G'
        else:
            display_message = "QR no detectado."
            color_code = 'R'

    elif key == ord('h'):
        clicked_points = []
        collecting_points = True
        display_message = "Seleccioná 4 puntos con clic. (Aborta con cualquier tecla)"

    elif key != 255 and collecting_points:
        # Tecla apretada mientras se clickea => abortar
        clicked_points = []
        collecting_points = False
        display_message = "Selección cancelada."
        color_code = 'R'

    elif len(clicked_points) == 4 and not collecting_points: # Se agregó la condición `not collecting_points` para que solo se ejecute una vez al terminar de recolectar los 4 puntos
        src_pts = np.float32(clicked_points)
        dst_pts = np.float32([[0, 0], [grid_size[0], 0], [grid_size[0], grid_size[1]], [0, grid_size[1]]])
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
        display_message = "Homografía a partir de puntos manuales calculada."
        color_code = 'G'
        clicked_points = [] # Se limpian los puntos después de calcular la homografía.

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()