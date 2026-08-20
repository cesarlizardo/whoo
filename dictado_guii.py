import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import customtkinter as ctk
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import os
import time
import threading
from datetime import datetime
from pynput import keyboard

# Configuración visual
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FRECUENCIA_MUESTREO = 44100
NOMBRE_ARCHIVO_AUDIO = "temp_dictado.wav"
CARPETA_DESTINO = os.path.expanduser("~/Desktop/Dictados")

class AppDictado(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Dictado por Voz")
        self.geometry("360x280")
        self.resizable(False, False)

        self.grabando = False
        self.frames = []
        self.stream = None
        self.inicio_grabacion = 0

        # Componentes visuales
        self.lbl_titulo = ctk.CTkLabel(self, text="DICTADO WHISPER", font=("Helvetica", 18, "bold"))
        self.lbl_titulo.pack(pady=(20, 5))

        self.lbl_estado = ctk.CTkLabel(self, text="En espera...", font=("Helvetica", 14), text_color="gray")
        self.lbl_estado.pack(pady=10)

        self.btn_grabar = ctk.CTkButton(
            self, 
            text="Iniciar Grabación", 
            fg_color="#28a745", 
            hover_color="#218838",
            font=("Helvetica", 14, "bold"),
            height=40,
            command=self.alternar_dictado
        )
        self.btn_grabar.pack(pady=15)

        self.lbl_info = ctk.CTkLabel(
            self, 
            text="Atajos: Ctrl+Alt+D (Grabar) | Ctrl+Alt+Q (Salir)", 
            font=("Helvetica", 11), 
            text_color="gray"
        )
        self.lbl_info.pack(side="bottom", pady=15)

        # Iniciar escuchador de teclado en segundo plano
        threading.Thread(target=self.iniciar_hotkeys, daemon=True).start()

    def callback_audio(self, indata, frames_count, time_info, status):
        if self.grabando:
            self.frames.append(indata.copy())

    def alternar_dictado(self):
        if not self.grabando:
            self.iniciar_grabacion()
        else:
            threading.Thread(target=self.detener_y_transcribir).start()

    def iniciar_grabacion(self):
        self.grabando = True
        self.frames = []
        self.inicio_grabacion = time.time()

        self.stream = sd.InputStream(
            samplerate=FRECUENCIA_MUESTREO,
            channels=1,
            dtype='int16',
            callback=self.callback_audio
        )
        self.stream.start()

        self.lbl_estado.configure(text="🎤 Grabando...", text_color="#dc3545")
        self.btn_grabar.configure(text="Detener y Transcribir", fg_color="#dc3545", hover_color="#c82333")

    def detener_y_transcribir(self):
        duracion = time.time() - self.inicio_grabacion
        self.grabando = False
        self.stream.stop()
        self.stream.close()

        self.lbl_estado.configure(text="⚙ Transcribiendo...", text_color="#ffc107")
        self.btn_grabar.configure(state="disabled")

        if not self.frames:
            self.lbl_estado.configure(text="En espera...", text_color="gray")
            self.btn_grabar.configure(state="normal", text="Iniciar Grabación", fg_color="#28a745")
            return

        audio_data = np.concatenate(self.frames, axis=0)
        wav.write(NOMBRE_ARCHIVO_AUDIO, FRECUENCIA_MUESTREO, audio_data)

        model = whisper.load_model("base")
        resultado = model.transcribe(NOMBRE_ARCHIVO_AUDIO, language="es", fp16=False)
        texto = resultado["text"].strip()

        if texto:
            os.makedirs(CARPETA_DESTINO, exist_ok=True)
            fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ruta = os.path.join(CARPETA_DESTINO, f"dictado_{fecha_hora}.txt")
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(texto)

        if os.path.exists(NOMBRE_ARCHIVO_AUDIO):
            os.remove(NOMBRE_ARCHIVO_AUDIO)

        self.lbl_estado.configure(text=f"✔ Guardado ({int(duracion)}s)", text_color="#28a745")
        self.btn_grabar.configure(state="normal", text="Iniciar Grabación", fg_color="#28a745")

    def salir_programa(self):
        self.destroy()
        os._exit(0)

    def iniciar_hotkeys(self):
        with keyboard.GlobalHotKeys({
            '<ctrl>+<alt>+d': self.alternar_dictado,
            '<ctrl>+<alt>+q': self.salir_programa
        }) as h:
            h.join()

if __name__ == "__main__":
    app = AppDictado()
    app.mainloop()
