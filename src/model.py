from pyannote.audio import Pipeline
import torch
import librosa

class DiarizationModel:
    def __init__(self, token):
        self.model = "pyannote/speaker-diarization-community-1"
        self.token = token
        self.pipeline = self._init_pipeline()

    def _init_pipeline(self):
        pipeline = Pipeline.from_pretrained(self.model, token=self.token)

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        return pipeline

    def process_audio(self, audio_file, **kwargs):
        # Загружаем аудио через librosa, получаем numpy array (время,)
        # и частоту дискретизации. Librosa автоматически приводит к моно.
        waveform_np, sample_rate = librosa.load(audio_file, sr=None, mono=True)
        # Преобразуем numpy -> torch tensor и добавляем размерность канала: (1, время)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, T)
        # Упаковываем в формат, понятный pyannote
        audio_batch = {"waveform": waveform, "sample_rate": sample_rate}
        diarization = self.pipeline(audio_batch, **kwargs)
        return diarization
