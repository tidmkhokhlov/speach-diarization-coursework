import subprocess
import tempfile
import os

def preprocess_audio(audio_file: str):
    """
    Преобразует аудиофайл в WAV, моно, 16 кГц с помощью FFmpeg.
    Возвращает путь к временному обработанному файлу.
    """
    try:
        # Создаём временный файл с расширением .wav
        fd, temp_wav = tempfile.mkstemp(suffix=".wav", prefix="preprocessed_")
        os.close(fd)

        # Команда FFmpeg
        cmd = [
            "ffmpeg",
            "-i", audio_file,
            "-ac", "1",          # моно
            "-ar", "16000",      # частота 16 кГц
            "-y",                # перезаписывать без вопросов
            temp_wav
        ]

        # Запускаем и ждём завершения
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_wav
    except Exception as e:
        print(f"Preprocessing error: \n\n{e}")