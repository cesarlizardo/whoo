import os
import sys
import ssl
import time
import threading
from datetime import datetime

# Ocultar el icono de cohete de Python en el Dock de macOS
if sys.platform == "darwin":
    try:
        import AppKit
        # NSApplicationActivationPolicyAccessory = 1 (mantiene la ventana pero oculta el icono del Dock)
        AppKit.NSApplication.sharedApplication().setActivationPolicy_(1)
    except Exception:
        pass

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ssl._create_default_https_context = ssl._create_unverified_context

import torch
torch.set_num_threads(1)

import customtkinter as ctk
import sounddevice as sd
import numpy as np
import whisper
from pynput import keyboard

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FRECUENCIA_MUESTREO = 16000
CARPETA_DESTINO = os.path.expanduser("~/Desktop/Whoo")

class AppWhoo(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Whoo")
        self.geometry("360x280")
        self.resizable(False, False)

        self.grabando = False
        self.frames = []
        self.stream = None
        self.inicio_grabacion = 0
        self.modelo = None
        self.hotkeys_listener = None

        os.makedirs(CARPETA_DESTINO, exist_ok=True)

        self.lbl_titulo = ctk.CTkLabel(self, text="WHOO", font=("Helvetica", 22, "bold"))
        self.lbl_titulo.pack(pady=(20, 5))

        self.lbl_estado = ctk.CTkLabel(self, text="⚙ Cargando modelo...", font=("Helvetica", 13), text_color="#ffc107")
        self.lbl_estado.pack(pady=10)

        self.btn_grabar = ctk.CTkButton(
            self, 
            text="Cargando...", 
            fg_color="#6c757d", 
            font=("Helvetica", 14, "bold"),
            height=40,
            state="disabled",
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

        self.iniciar_hotkeys()
        threading.Thread(target=self.cargar_modelo, daemon=True).start()

    def iniciar_hotkeys(self):
        try:
            self.hotkeys_listener = keyboard.GlobalHotKeys({
                '<ctrl>+<alt>+d': lambda: self.after(0, self.alternar_dictado),
                '<ctrl>+<alt>+q': lambda: self.after(0, self.salir_programa)
            })
            self.hotkeys_listener.start()
        except Exception as e:
            print(f"[!] No se pudieron activar atajos globales: {e}")

    def cargar_modelo(self):
        try:
            self.modelo = whisper.load_model("tiny")
            self.after(0, lambda: self.lbl_estado.configure(text="En espera...", text_color="gray"))
            self.after(0, lambda: self.btn_grabar.configure(text="Iniciar Grabación", fg_color="#28a745", state="normal"))
        except Exception as e:
            self.registrar_error("Error cargando modelo", e)

    def callback_audio(self, indata, frames_count, time_info, status):
        if self.grabando:
            self.frames.append(indata.copy())

    def alternar_dictado(self):
        if self.modelo is None:
            return
            
        if not self.grabando:
            self.iniciar_grabacion()
        else:
            threading.Thread(target=self.detener_y_transcribir, daemon=True).start()

    def iniciar_grabacion(self):
        self.grabando = True
        self.frames = []
        self.inicio_grabacion = time.time()

        try:
            self.stream = sd.InputStream(
                samplerate=FRECUENCIA_MUESTREO,
                channels=1,
                dtype='float32',
                callback=self.callback_audio
            )
            self.stream.start()

            self.lbl_estado.configure(text="🎤 Grabando...", text_color="#dc3545")
            self.btn_grabar.configure(text="Detener y Transcribir", fg_color="#dc3545", hover_color="#c82333")
        except Exception as e:
            self.registrar_error("Error de Micrófono", e)

    def detener_y_transcribir(self):
        duracion = time.time() - self.inicio_grabacion
        self.grabando = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

        self.after(0, lambda: self.lbl_estado.configure(text="⚙ Transcribiendo...", text_color="#ffc107"))
        self.after(0, lambda: self.btn_grabar.configure(state="disabled"))

        try:
            if not self.frames:
                self.after(0, lambda: self.lbl_estado.configure(text="En espera...", text_color="gray"))
                return

            audio_data = np.concatenate(self.frames, axis=0).flatten()
            resultado = self.modelo.transcribe(audio_data, language="es", fp16=False)
            texto = resultado["text"].strip()

            if texto:
                fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                ruta = os.path.join(CARPETA_DESTINO, f"whoo_{fecha_hora}.txt")
                with open(ruta, "w", encoding="utf-8") as f:
                    f.write(texto)
                self.after(0, lambda: self.lbl_estado.configure(text=f"✔ Guardado ({int(duracion)}s)", text_color="#28a745"))
            else:
                self.after(0, lambda: self.lbl_estado.configure(text="⚠ Sin texto detectado", text_color="#ffc107"))

        except Exception as e:
            self.registrar_error("Error transcripción", e)

        finally:
            self.after(0, lambda: self.btn_grabar.configure(state="normal", text="Iniciar Grabación", fg_color="#28a745"))

    def registrar_error(self, mensaje_ui, error):
        log_path = os.path.join(CARPETA_DESTINO, "error_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {mensaje_ui}: {str(error)}\n")
        self.after(0, lambda: self.lbl_estado.configure(text=f"❌ {mensaje_ui}", text_color="#dc3545"))

    def salir_programa(self):
        if self.hotkeys_listener:
            try:
                self.hotkeys_listener.stop()
            except Exception:
                pass
        self.destroy()
        os._exit(0)

if __name__ == "__main__":
    app = AppWhoo()
    app.mainloop()
