from diarize import diarize
from .interface import DiarizationInterface

class DiarizenDiarization(DiarizationInterface):
    def get_name(self) -> str:
        return "DiariZen (CPU-only, fast)"

    def diarize(self, audio_filepath: str, **kwargs):
        # Извлекаем параметры, которые поддерживает diarize
        min_speakers = kwargs.get('min_speakers')
        max_speakers = kwargs.get('max_speakers')
        num_speakers = kwargs.get('num_speakers')

        result = diarize(
            audio_filepath,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers
        )
        # Библиотека diarize возвращает объект с атрибутом .segments
        # Каждый сегмент имеет .start, .end, .speaker (int)
        return result