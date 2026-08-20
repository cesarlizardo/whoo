import os
import sys
import ssl
import time
import threading
import traceback
from datetime import datetime
from PIL import Image

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
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))

class AppWhoo(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Whoo")
        self.geometry("360x450")
        self.resizable(False, False)

        self.establecer_icono_dock()

        self.grabando = False
        self.frames = []
        self.stream = None
        self.inicio_grabacion = 0
        self.modelo = None
        self.hotkeys_listener = None
        self.device_mapping = {}

        os.makedirs(CARPETA_DESTINO, exist_ok=True)

        # Logo del Buho
        ruta_logo = os.path.join(RUTA_BASE, "logo.png")
        if os.path.exists(ruta_logo):
            img_buho = Image.open(ruta_logo)
            self.logo_img = ctk.CTkImage(light_image=img_buho, dark_image=img_buho, size=(100, 100))
            self.lbl_logo = ctk.CTkLabel(self, image=self.logo_img, text="")
            self.lbl_logo.pack(pady=(15, 2))

        self.lbl_titulo = ctk.CTkLabel(self, text="WHOO", font=("Helvetica", 20, "bold"))
        self.lbl_titulo.pack(pady=(0, 2))

        # Seleccion de Microfono
        self.lbl_mic = ctk.CTkLabel(self, text="ENTRADA DE AUDIO", font=("Helvetica", 10, "bold"), text_color="#888888")
        self.lbl_mic.pack(pady=(5, 2))

        self.combo_mic = ctk.CTkOptionMenu(
            self,
            values=["Cargando micrófonos..."],
            font=("Helvetica", 11),
            dropdown_font=("Helvetica", 11),
            fg_color="#2b2b2b",
            button_color="#3d3d3d",
            button_hover_color="#4d4d4d",
            width=280,
            height=28
        )
        self.combo_mic.pack(pady=(0, 10))

        self.lbl_estado = ctk.CTkLabel(self, text="CARGANDO MODELO...", font=("Helvetica", 11, "bold"), text_color="#aaaaaa")
        self.lbl_estado.pack(pady=5)

        self.btn_grabar = ctk.CTkButton(
            self, 
            text="CARGANDO", 
            fg_color="#333333", 
            hover_color="#444444",
            text_color="#ffffff",
            font=("Helvetica", 12, "bold"),
            height=38,
            corner_radius=6,
            state="disabled",
            command=self.alternar_dictado
        )
        self.btn_grabar.pack(pady=10)

        self.lbl_info = ctk.CTkLabel(
            self, 
            text="Atajos: Ctrl+Alt+D (Grabar) | Ctrl+Alt+Q (Salir)", 
            font=("Helvetica", 10), 
            text_color="#666666"
        )
        self.lbl_info.pack(side="bottom", pady=12)

        self.obtener_dispositivos_audio()
        self.iniciar_hotkeys()
        threading.Thread(target=self.cargar_modelo, daemon=True).start()

    def obtener_dispositivos_audio(self):
        try:
            dispositivos = sd.query_devices()
            default_in = sd.default.device[0]
            
            nombres_mic = []
            self.device_mapping = {}
            default_name = None

            for idx, dev in enumerate(dispositivos):
                if dev['max_input_channels'] > 0:
                    nombre = f"{dev['name']}"
                    if list(dispositivos).count(dev) > 1 or any(nombre in k for k in self.device_mapping.keys()):
                        display_name = f"{nombre} [{idx}]"
                    else:
                        display_name = nombre
                    
                    nombres_mic.append(display_name)
                    self.device_mapping[display_name] = idx

                    if idx == default_in:
                        default_name = display_name

            if nombres_mic:
                self.combo_mic.configure(values=nombres_mic)
                if default_name and default_name in nombres_mic:
                    self.combo_mic.set(default_name)
                else:
                    self.combo_mic.set(nombres_mic[0])
            else:
                self.combo_mic.configure(values=["Sin micrófonos"])
                self.combo_mic.set("Sin micrófonos")
        except Exception as e:
            traceback.print_exc()
            self.combo_mic.configure(values=["Error al listar mics"])
            self.combo_mic.set("Error al listar mics")

    def establecer_icono_dock(self):
        if sys.platform == "darwin":
            try:
                import AppKit
                from Cocoa import NSImage
                ruta_logo = os.path.join(RUTA_BASE, "logo.png")
                if os.path.exists(ruta_logo):
                    ns_img = NSImage.alloc().initWithContentsOfFile_(ruta_logo)
                    AppKit.NSApplication.sharedApplication().setApplicationIconImage_(ns_img)
            except Exception:
                pass

    def iniciar_hotkeys(self):
        try:
            self.hotkeys_listener = keyboard.GlobalHotKeys({
                '<ctrl>+<alt>+d': lambda: self.after(0, self.alternar_dictado),
                '<ctrl>+<alt>+q': lambda: self.after(0, self.salir_programa)
            })
            self.hotkeys_listener.start()
        except Exception:
            pass

    def cargar_modelo(self):
        try:
            self.modelo = whisper.load_model("base")
            self.after(0, lambda: self.lbl_estado.configure(text="EN ESPERA", text_color="#aaaaaa"))
            self.after(0, lambda: self.btn_grabar.configure(text="INICIAR GRABACION", fg_color="#1b5e20", hover_color="#2e7d32", state="normal"))
        except Exception as e:
            traceback.print_exc()
            self.registrar_error("ERROR AL CARGAR MODELO", e)

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

        selected_mic = self.combo_mic.get()
        device_idx = self.device_mapping.get(selected_mic, None)

        try:
            self.stream = sd.InputStream(
                samplerate=FRECUENCIA_MUESTREO,
                channels=1,
                dtype='float32',
                device=device_idx,
                callback=self.callback_audio
            )
            self.stream.start()

            self.combo_mic.configure(state="disabled")
            self.lbl_estado.configure(text="GRABANDO...", text_color="#d32f2f")
            self.btn_grabar.configure(text="DETENER Y TRANSCRIBIR", fg_color="#b71c1c", hover_color="#c62828")
        except Exception as e:
            traceback.print_exc()
            self.grabando = False
            self.combo_mic.configure(state="normal")
            self.registrar_error("ERROR DE MICROFONO", e)

    def detener_y_transcribir(self):
        duracion = time.time() - self.inicio_grabacion
        self.grabando = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

        self.after(0, lambda: self.combo_mic.configure(state="normal"))
        self.after(0, lambda: self.lbl_estado.configure(text="TRANSCRIBIENDO...", text_color="#f57c00"))
        self.after(0, lambda: self.btn_grabar.configure(state="disabled", fg_color="#333333"))

        try:
            if not self.frames:
                self.after(0, lambda: self.lbl_estado.configure(text="EN ESPERA", text_color="#aaaaaa"))
                return

            audio_data = np.concatenate(self.frames, axis=0).flatten().astype(np.float32)

            if len(audio_data) < 8000:
                self.after(0, lambda: self.lbl_estado.configure(text="GRABACION MUY CORTA", text_color="#f57c00"))
                return

            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = (audio_data / max_val) * 0.95

            resultado = self.modelo.transcribe(
                audio_data,
                language="es",
                fp16=False,
                temperature=0.0,
                beam_size=5,
                best_of=5,
                patience=1.0,
                condition_on_previous_text=False,
                initial_prompt="Transcripción exacta, limpia y fluida en español, respetando la puntuación, ortografía, mayúsculas y números.",
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            texto = resultado["text"].strip()

            if texto:
                fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                ruta = os.path.join(CARPETA_DESTINO, f"whoo_{fecha_hora}.txt")
                with open(ruta, "w", encoding="utf-8") as f:
                    f.write(texto)
                self.after(0, lambda: self.lbl_estado.configure(text=f"GUARDADO ({int(duracion)}s)", text_color="#388e3c"))
            else:
                self.after(0, lambda: self.lbl_estado.configure(text="SIN TEXTO DETECTADO", text_color="#f57c00"))

        except Exception as e:
            traceback.print_exc()
            self.registrar_error("ERROR DE TRANSCRIPCION", e)

        finally:
            self.after(0, lambda: self.btn_grabar.configure(state="normal", text="INICIAR GRABACION", fg_color="#1b5e20", hover_color="#2e7d32"))

    def registrar_error(self, mensaje_ui, error):
        log_path = os.path.join(CARPETA_DESTINO, "error_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {mensaje_ui}: {str(error)}\n")
        self.after(0, lambda: self.lbl_estado.configure(text=mensaje_ui, text_color="#d32f2f"))

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
