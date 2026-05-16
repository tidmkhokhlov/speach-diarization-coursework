from abc import ABC, abstractmethod

class DiarizationInterface(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """Возвращает название модели для интерфейса."""
        pass

    @abstractmethod
    def diarize(self, audio_filepath: str, **kwargs):
        """
        Основной метод для получения диаризации.

        Args:
            audio_filepath (str): Путь к аудиофайлу.
            **kwargs: Дополнительные параметры (min_speakers, max_speakers).

        Returns:
            Объект, совместимый с pyannote.core.Annotation (или его аналог).
            Для простоты можно возвращать список сегментов или сам объект Annotation.
        """
        pass