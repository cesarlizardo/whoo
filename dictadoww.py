import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import os
import time
from datetime import datetime
from pynput import keyboard

FRECUENCIA_MUESTREO = 44100
NOMBRE_ARCHIVO_AUDIO = "temp_dictado.wav"
CARPETA_DESTINO = os.path.expanduser("~/Desktop/Dictados")

BANNER_LOGO = r"""
======================================================
  ____  _  ____ _____  _    ____   ___  ____ _____ 
 |  _ \| |/ ___|_   _|/ \  |  _ \ / _ \/ ___|_   _|
 | | | | | |     | | / _ \ | | | | | | \___ \ | |  
 | |_| | | |___  | |/ ___ \| |_| | |_| |___) || |  
 |____/|_|\____| |_/_/   \_\____/ \___/|____/ |_|  
                                                   
               DICTADO POR VOZ - WHISPER
======================================================
"""

grabando = False
frames = []
stream = None
TIEMPO_INICIO_SESION = time.time()
inicio_grabacion = 0

def callback_audio(indata, frames_count, time_info, status):
    if grabando:
        frames.append(indata.copy())

def formatear_tiempo(segundos):
    minutos = int(segundos // 60)
    segs = int(segundos % 60)
    if minutos > 0:
        return f"{minutos}m {segs}s"
    return f"{segs}s"

def guardar_texto_en_escritorio(texto):
    if not texto:
        print("[!] No se generó archivo porque no se detectó voz.")
        return

    os.makedirs(CARPETA_DESTINO, exist_ok=True)
    fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ruta_archivo = os.path.join(CARPETA_DESTINO, f"dictado_{fecha_hora}.txt")

    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write(texto)

    print(f"[+] Transcripción guardada en:\n    {ruta_archivo}")

def alternar_dictado():
    global grabando, frames, stream, inicio_grabacion

    if not grabando:
        grabando = True
        frames = []
        inicio_grabacion = time.time()
        
        stream = sd.InputStream(
            samplerate=FRECUENCIA_MUESTREO, 
            channels=1, 
            dtype='int16', 
            callback=callback_audio
        )
        stream.start()
        
        print("\n[+] GRABANDO AUTOMÁTICAMENTE... Habla el tiempo que necesites.")
        print("[-] Presiona (Ctrl + Alt + D) para DETENER y transcribir.")

    else:
        duracion_dictado = time.time() - inicio_grabacion
        grabando = False
        stream.stop()
        stream.close()
        
        print(f"\n[OK] Grabación detenida. Tiempo grabado: {formatear_tiempo(duracion_dictado)}")

        if not frames:
            print("[!] No se capturó ningún audio.")
            return

        audio_data = np.concatenate(frames, axis=0)
        wav.write(NOMBRE_ARCHIVO_AUDIO, FRECUENCIA_MUESTREO, audio_data)

        print("[*] Transcribiendo...")
        inicio_transcripcion = time.time()
        model = whisper.load_model("base")
        resultado = model.transcribe(
            NOMBRE_ARCHIVO_AUDIO, 
            language="es", 
            fp16=False, 
            no_speech_threshold=0.6
        )
        tiempo_transcripcion = time.time() - inicio_transcripcion
        texto = resultado["text"].strip()

        print("="*30)
        print("TEXTO DETECTADO:")
        print(texto if texto else "(Sin voz clara)")
        print("="*30)
        print(f"[*] Duración del dictado: {formatear_tiempo(duracion_dictado)}")
        print(f"[*] Tiempo de procesado: {formatear_tiempo(tiempo_transcripcion)}")

        guardar_texto_en_escritorio(texto)

        if os.path.exists(NOMBRE_ARCHIVO_AUDIO):
            os.remove(NOMBRE_ARCHIVO_AUDIO)

        print("\n[*] En espera...")
        print("     • (Ctrl + Alt + D) -> Grabar de nuevo")
        print("     • (Ctrl + Alt + Q) -> Salir a la terminal")

def salir_del_programa():
    duracion_total = time.time() - TIEMPO_INICIO_SESION
    print(f"\n[*] Tiempo total de la sesión activa: {formatear_tiempo(duracion_total)}")
    print("[*] Cerrando el script. ¡Hasta luego!")
    os._exit(0)

if __name__ == "__main__":
    print(BANNER_LOGO)
    print("[+] Script de dictado activo.")
    print("[*] Ctrl + Alt + D  -->  DETENER / REINICIAR grabación")
    print("[*] Ctrl + Alt + Q  -->  SALIR a la terminal")
    
    alternar_dictado()

    with keyboard.GlobalHotKeys({
        '<ctrl>+<alt>+d': alternar_dictado,
        '<ctrl>+<alt>+q': salir_del_programa
    }) as h:
        h.join()
