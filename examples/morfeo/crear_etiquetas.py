import csv
from datetime import datetime
import os
import threading

import cv2
import numpy as np

# Variables para almacenar puntos de clic y posición del ratón
click_points = []
current_mouse_pos = (0, 0)
drawing = False
rectangle_just_drawn = False
finished_videos = {}
key = 0
csv_filename = 'rectangulos_videos.csv'
original_frame = None
stable_frame = None
video_filename = None
reset = False

class Rectangulo:
    lowX = 0
    lowY = 0
    highX = 0
    highY = 0

# Función para encontrar el primer video AVI en la carpeta que no comience con 'HECHO'
def get_new_avi_video(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('.avi') \
            and not file.startswith('HECHO')\
            and not file.startswith('EXTRAIDO'):
            return os.path.join(folder_path, file)
    return None

def update_video_csv():
    file_exists = False
    try:
        with open(csv_filename, 'r', newline='', encoding='utf-8') as file:
            file_exists = True
    except FileNotFoundError:
        print(f"Este vídeo no existe: {csv_filename}")
    
    # Open the CSV file in append mode
    with open(csv_filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header if the file is new
        if not file_exists:
            writer.writerow(['titulo_video', 'rectangulos_semáforo', 'puntos', 'fecha'])

        # Iterate through the finished_videos dictionary and write each entry
        for video_path, (num_rectangles, click_points) in finished_videos.items():
            fecha = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get the current date and time
            writer.writerow([video_path, num_rectangles, str(click_points), fecha])

# Manejador de eventos del ratón
def mouse_event(event, x, y, flags, param):
    global click_points, current_mouse_pos, drawing, rectangle_just_drawn, stable_frame, reset
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(click_points)%2 == 0:
            # Primer clic para el primer punto del rectángulo
            click_points.append((x, y))
            drawing = True
        elif len(click_points)%2 == 1:
            # Cierra el rectángulo
            # Finalizar el dibujo del rectángulo después del segundo clic
            modified_second_point = adjutst_rectangle(click_points[-1], (x,y))
            click_points.append(modified_second_point)
            rectangle_just_drawn = True
            drawing = False
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_mouse_pos = (x, y)
        drawing = True
    elif event == cv2.EVENT_RBUTTONDOWN and len(click_points) > 0:
        click_points.pop()
        reset = True
        if len(click_points) % 2 == 1:
            drawing = True
        else:
            drawing = False
        print(f"click points: {click_points}")

def pad_and_resize(image, target_size=(25, 75)):
    # Original dimensions
    h, w = image.shape[:2]
    target_ratio = 1 / 3  # width:height ratio

    # Calculate target height and width
    if w / h > target_ratio:
        new_h = int(w / target_ratio)
        new_w = w
    else:
        new_w = int(h * target_ratio)
        new_h = h

    # Calculate padding
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left

    # Add padding
    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # Resize to target dimensions
    resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

def draw_stable_rectangles(frame):
    for i in range(len(click_points)-1):
        if i % 2:
            continue
        frame = cv2.rectangle(frame, click_points[i], click_points[i+1], (255, 0, 0), 2)
    return frame

def adjutst_rectangle(point1, point2):
    """
    Increases the rectangle moving the bottom-right edge to get a 1:3 ratio rectangle
    Parameters:
    - point1 (tuple): top left
    - point2 (tuple): bottom right
    Returns:
    - new_point2 (tuple)
    """
    lowX, lowY = point1
    highX, highY = point2
    width = highX - lowX
    height = highY - lowY
    ratio = width / (height + 1e-6)
    if ratio < 1/3:  # Aumentamos ancho
        new_width = max(height / 3, 0)
        new_height = height
    elif ratio > 1/3:
        new_width = width
        new_height = max(width * 3, 0)
    else:
        print("That has the right ratio")
        new_height = height
        new_width = width
        
    new_point2 = (lowX + int(new_width), lowY + int(new_height))
    return new_point2

def guess_light_on(image, num_divisions=3):
    """
    Divide the image into num_divisions horizontal parts.
    Guess which one of the three is on
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]
    height, width = v_channel.shape[:2]
    division_height = height // num_divisions
    divisions = []
    for i in range(num_divisions):
        lowY = i * division_height
        highY = (i + 1) * division_height
        current_division = v_channel[lowY:highY, :]
        divisions.append(np.mean(current_division))
    [print(x) for x in divisions]
    cv2.imshow("divisions", v_channel)
    cv2.waitKey(1)
    light_on = np.argmax(divisions)
    if light_on == 0:
        light = 'rojo'
    else:
        light = 'norojo'
    return light

def crop_video(video_path, points, output_folder):
    """
    Crops a video at specified points and saves each cropped section as a new video.

    Parameters:
    - video_path (str): Path to the video file.
    - points (list): List of tuples containing rectangle points (x, y).
        topleft - bottomright
    - output_folder (str): Folder to save cropped videos.
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return

    # Create output folders if they don't exist
    rojo_folder = os.path.join(output_folder, 'rojo')
    norojo_folder = os.path.join(output_folder, 'norojo')
    for folder in [output_folder, rojo_folder, norojo_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Process each rectangle
    rectangulos = []
    for i, (x, y) in enumerate(points):
        rectangulo = Rectangulo()
        if i%2 == 0:  # A rectangle each two points
            x1 = x
            y1 = y
            continue
        else:
            x2 = x
            y2 = y
        rectangulo.lowX = min(x1, x2)
        rectangulo.highX = max(x1, x2)
        rectangulo.lowY = min(y1, y2)
        rectangulo.highY = max(y1, y2)
        rectangulos.append(rectangulo)


    frame_number = 0
    while True:
        # Read the current frame
        ret, frame = cap.read()
        if not ret:
            break

        for i, semaforo in enumerate(rectangulos):
            current_semaforo = frame[semaforo.lowY : semaforo.highY, semaforo.lowX : semaforo.highX]
            color = guess_light_on(current_semaforo)
            image_name = f"{os.path.basename(video_path).split('.')[0]}_frame{frame_number}_semaforo{i}.jpg"
            image_path = os.path.join(output_folder, color, image_name)
            cv2.imwrite(image_path, current_semaforo)
        frame_number += 1
    print(f"Витягнуто {frame_number} кадрів. Кадри збережені у: {output_folder}")
        
    # Release the video capture object
    cap.release()

# Programa principal
folder_path = r"C:\chapus\fotorrojo"

while key != 27:
    video_path = get_new_avi_video(folder_path)
    video_filename = os.path.basename(video_path)

    if video_path:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        original_frame = frame.copy()
        temporal_frame = frame.copy()
        stable_frame = frame.copy()

        if ret:
            # Mostrar el primer fotograma
            print("Привіт друже\n")
            cv2.imshow('Video ' + video_filename, frame)
            cv2.setMouseCallback('Video ' + video_filename, mouse_event)
            
            # Bucle para actualizar la posición del rectángulo
            while True:
                key = cv2.waitKey(50)  # Esperar 100ms
                if key == 27:  # Salir con ESC
                    update_video_csv()
                    break
                elif key == 13:  # Guardar rectángulos
                    next_video = True
                    finished_videos[video_filename] = (len(click_points), click_points)
                    update_video_csv()
                    today_string = datetime.now().strftime('%y%m%d') + '_'
                    video_to_extract = os.path.join(os.path.dirname(video_path), 'EXTRAIDO_' + today_string + video_filename)
                    print(f"Відео {video_filename} збережено з {len(click_points)//2} світлофорами")

                    # Liberamos el vídeo y lo extraemos
                    cap.release()
                    os.rename(video_path, video_to_extract)
                    #thread = threading.Thread(target=crop_video, args=(video_to_extract, click_points, video_to_extract.split('.')[0]))
                    #thread.start()
                    crop_video(video_to_extract, click_points, video_to_extract.split('.')[0])
                    click_points = []
                    break

                if len(click_points) == 0 or reset:
                    stable_frame = draw_stable_rectangles(original_frame.copy())
                
                # Dibujar el rectángulo dinámicamente si el nº de clics es impar
                if drawing and (len(click_points)%2 == 1):
                    temporal_frame = stable_frame.copy()
                    corrected_point2 = adjutst_rectangle(click_points[-1], current_mouse_pos)
                    cv2.rectangle(temporal_frame, click_points[-1], corrected_point2, (255, 0, 0), 2)
                # Añadir rectángulo recién terminado a la imagen
                elif not drawing and rectangle_just_drawn:
                    stable_frame = draw_stable_rectangles(stable_frame)
                    temporal_frame = stable_frame.copy()
                    rectangle_just_drawn = False
                    print("Прямокутник збережено. Додайте ще один або натисніть Enter, щоб перейти до наступного відео.")
                else:
                    temporal_frame = stable_frame.copy()
                
                cv2.imshow('Video ' + video_filename, temporal_frame)
                cv2.waitKey(1)

        else:
            print("No se pudo leer el primer fotograma del video.")

        cv2.destroyWindow('Video ' + video_filename)
        cap.release()
    else:
        print(f"No se encontró un video AVI válido en la carpeta {folder_path}.")

cv2.destroyAllWindows()
