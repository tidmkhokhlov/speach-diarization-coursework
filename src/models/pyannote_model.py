from pyannote.audio import Pipeline
import torch
import librosa
from .interface import DiarizationInterface

class PyannoteDiarization(DiarizationInterface):
    def __init__(self, token: str):
        self.model_name = "pyannote/speaker-diarization-community-1"
        self.token = token
        self.pipeline = self._init_pipeline()

    def _init_pipeline(self):
        pipeline = Pipeline.from_pretrained(self.model_name, token=self.token)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        return pipeline

    def get_name(self) -> str:
        return "Pyannote (community-1)"

    def diarize(self, audio_filepath: str, **kwargs):
        # Загружаем аудио через librosa (обход torchcodec)
        waveform_np, sample_rate = librosa.load(audio_filepath, sr=None, mono=True)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, T)
        audio_batch = {"waveform": waveform, "sample_rate": sample_rate}
        return self.pipeline(audio_batch, **kwargs)