from config import settings
from model import DiarizationModel
from utils import preprocess_audio

class DiarizationService:
    def __init__(self):
        self.model = DiarizationModel(settings.HF_TOKEN)

    def diarize(self, audio_file, **kwargs):
        formatted_audio = preprocess_audio(audio_file)
        return self.model.process_audio(formatted_audio, **kwargs)