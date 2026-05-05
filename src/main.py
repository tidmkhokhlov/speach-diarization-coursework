import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from service import DiarizationService
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

service = DiarizationService()

def diarize_with_plot(audio_filepath, min_speakers, max_speakers):
    # Передаём параметры (даже если модель их не поддерживает – не страшно)
    diarization = service.diarize(audio_filepath, min_speakers=min_speakers, max_speakers=max_speakers)

    # Извлекаем сегменты из атрибута speaker_diarization
    records = []
    for turn, speaker in diarization.speaker_diarization:
        records.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
            "duration": round(turn.end - turn.start, 2)
        })

    df = pd.DataFrame(records)
    if df.empty:
        return "Нет сегментов речи", None

    # Строим таймлайн
    fig, ax = plt.subplots(figsize=(12, 3))
    speakers = df["speaker"].unique()
    speaker_to_num = {sp: i for i, sp in enumerate(speakers)}
    df["speaker_num"] = df["speaker"].map(speaker_to_num)

    for _, row in df.iterrows():
        ax.barh(
            y=row["speaker_num"],
            width=row["duration"],
            left=row["start"],
            height=0.8,
            color="skyblue" if row["speaker_num"] % 2 == 0 else "lightcoral",
            edgecolor="black",
            linewidth=0.5
        )

    ax.set_yticks(list(speaker_to_num.values()))
    ax.set_yticklabels(list(speaker_to_num.keys()))
    ax.set_xlabel("Время (секунды)")
    ax.set_title("Результат диаризации (community-1)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    return df, fig

# Создаём интерфейс
with gr.Blocks(title="Диаризация речи", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🗣️ Демо диаризации речи (pyannote community-1)")
    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Загрузите или запишите аудио"
            )
            with gr.Row():
                min_sp = gr.Number(label="Минимум спикеров", value=1, precision=0, step=1)
                max_sp = gr.Number(label="Максимум спикеров", value=4, precision=0, step=1)
            run_btn = gr.Button("Распознать дикторов", variant="primary")
        with gr.Column(scale=3):
            output_table = gr.Dataframe(
                label="Сегменты речи",
                headers=["start", "end", "speaker", "duration"],
                interactive=False
            )
            output_plot = gr.Plot(label="Таймлайн")

    run_btn.click(
        fn=diarize_with_plot,
        inputs=[audio_input, min_sp, max_sp],
        outputs=[output_table, output_plot]
    )

    gr.Markdown(
        """
        **Примечания:**
        - Модель: `pyannote/speaker-diarization-community-1`
        - Аудио автоматически конвертируется в WAV, моно, 16 кГц.
        - Если разговор на русском языке, качество может быть ниже.
        """
    )

if __name__ == "__main__":
    demo.launch(debug=True)