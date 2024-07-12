from tkinter import *
from PIL import Image, ImageTk  # pip install Pillow
import sys
import cv2  # pip install opencv-contrib-python
from ultralytics import YOLO  # pip install ultralytics
import numpy as np
import pyttsx3  # pip install pyttsx3
import speech_recognition as sr  # pip install SpeechRecognition
import threading

# Configuración del asistente de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidad de la voz
engine.setProperty('volume', 1)  # Volumen de la voz

def speak(text):
    print(text)  # Imprimir en consola
    engine.say(text)
    engine.runAndWait()

def on_closing():
    speak("Cerrando el programa. Adiós!")
    root.quit()
    cap.release()
    root.destroy()

def recognize_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    with microphone as source:
        print("Escuchando...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio, language='es-ES')
        print("Comando reconocido:", command)
        return command.lower()
    except sr.UnknownValueError:
        print("No se entendió el comando")
        return ""
    except sr.RequestError:
        print("Error en el servicio de reconocimiento de voz")
        return ""

def process_command(command):
    if "oye naira" in command or "oye nayra" in command:
        speak("Hola, soy Naira tu asistente virtual, ¿qué se te ofrece?")
        command = recognize_speech()
        if "qué es lo que ves" in command:
            describe_scene()
        elif "adios naira" in command or "adios nayra" in command:
            on_closing()
        elif "reducir velocidad de voz" in command:
            reduce_voice_speed()
        elif "aumentar volumen de voz" in command:
            increase_voice_volume()
        else:
            speak("Comando no reconocido.")
    elif "qué es lo que ves" in command:
        describe_scene()

def describe_scene():
    if object_counts:
        for label, count in object_counts.items():
            speak(f"Veo {count} {label if count == 1 else label + 's'}")
    else:
        speak("No veo nada")

def reduce_voice_speed():
    current_rate = engine.getProperty('rate')
    new_rate = max(current_rate - 50, 50)
    engine.setProperty('rate', new_rate)
    speak("La velocidad de la voz ha sido reducida.")

def increase_voice_volume():
    current_volume = engine.getProperty('volume')
    new_volume = min(current_volume + 0.1, 1.0)
    engine.setProperty('volume', new_volume)
    speak("El volumen de la voz ha sido aumentado.")

def callback():
    global object_counts, results, frame_width, frame_count
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (640, 480))
            frame_width = frame.shape[1]
            results = model.predict(frame, stream=True, verbose=False)
            
            object_counts = {}
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    r = box.xyxy[0]
                    cls = box.cls
                    label_text = classes[int(cls)]
                    color = tuple(map(int, COLORS[int(cls)]))
                    cv2.rectangle(frame, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, 2)
                    cv2.putText(frame, label_text, (int(r[0]), int(r[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    if label_text in object_counts:
                        object_counts[label_text] += 1
                    else:
                        object_counts[label_text] = 1
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.config(image=imgtk)
    
    root.after(10, callback)

def voice_callback():
    threading.Thread(target=handle_voice_command, daemon=True).start()
    root.after(3000, voice_callback)

def handle_voice_command():
    command = recognize_speech()
    if command:
        process_command(command)

def load_model():
    global model
    model = YOLO("yolov8n.pt")

# Iniciar carga del modelo en un hilo separado
model_thread = threading.Thread(target=load_model, daemon=True)
model_thread.start()

# Cargar nombres de clases y asignar colores
with open("coco.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(1)
if cap.isOpened():
    print("Cámara iniciada")
else:
    sys.exit("Cámara desconectada")

# Diseño de la HMI
root = Tk()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.title("Visión Artificial")
camera_label = Label(root)
camera_label.grid(row=1, padx=20, pady=20)

object_counts = {}
results = []
frame_width = 0
frame_skip = 2  # Procesar cada segundo frame
frame_count = 0

root.after(10, callback)
root.after(3000, voice_callback)  # Ajustar el intervalo para evitar múltiples hilos simultáneos
root.mainloop()
