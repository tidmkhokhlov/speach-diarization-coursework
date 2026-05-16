import subprocess
import tempfile
import os

def preprocess_audio(audio_file: str):
    """
    Преобразует аудиофайл в WAV, моно, 16 кГц с помощью FFmpeg.
    Возвращает путь к временному обработанному файлу.
    """
    try:
        fd, temp_wav = tempfile.mkstemp(suffix=".wav", prefix="preprocessed_")
        os.close(fd)

        cmd = [
            "ffmpeg",
            "-i", audio_file,
            "-ac", "1",
            "-ar", "16000",
            "-y",
            temp_wav
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_wav
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return audio_file  # fallback