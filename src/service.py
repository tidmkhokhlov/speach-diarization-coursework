from config import settings
from models.pyannote_model import PyannoteDiarization
from models.diarizen_model import DiarizenDiarization
from utils import preprocess_audio

class DiarizationService:
    def __init__(self):
        self.models = {
            "pyannote": PyannoteDiarization(settings.HF_TOKEN),
            "diarizen": DiarizenDiarization()
        }

    def get_model_names(self):
        """Возвращает список кортежей (ключ, отображаемое_имя)."""
        return [(key, model.get_name()) for key, model in self.models.items()]

    def diarize(self, audio_file: str, model_key: str, **kwargs):
        selected_model = self.models.get(model_key)
        if not selected_model:
            raise ValueError(f"Unknown model key: {model_key}")

        formatted_audio = preprocess_audio(audio_file)
        return selected_model.diarize(formatted_audio, **kwargs)