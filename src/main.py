import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from service import DiarizationService
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

service = DiarizationService()

# Получаем список моделей в виде (ключ, отображаемое_имя)
model_items = service.get_model_names()  # [('pyannote', 'Pyannote (community-1)'), ('diarizen', 'DiariZen (CPU-only, fast)')]
model_keys = [key for key, _ in model_items]
model_labels = [label for _, label in model_items]
key_by_label = {label: key for key, label in model_items}

def diarize_with_plot(audio_filepath, min_speakers, max_speakers, model_label):
    if not audio_filepath:
        return pd.DataFrame(), None

    # Преобразуем выбранную метку в ключ модели
    model_key = key_by_label[model_label]

    result = service.diarize(audio_filepath, model_key, min_speakers=min_speakers, max_speakers=max_speakers)

    records = []
    if model_key == "pyannote":
        for turn, speaker in result.speaker_diarization:
            records.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker,
                "duration": round(turn.end - turn.start, 2)
            })
    elif model_key == "diarizen":
        for seg in result.segments:
            records.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "speaker": f"SPEAKER_{seg.speaker}",
                "duration": round(seg.end - seg.start, 2)
            })

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(), None

    # --- Построение таймлайна ---
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
    ax.set_title(f"Результат диаризации – {model_key}")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    return df, fig

# --- Интерфейс Gradio ---
with gr.Blocks(title="Диаризация речи") as demo:
    gr.Markdown("## 🗣️ Сравнение моделей диаризации речи")
    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Загрузите или запишите аудио"
            )
            # Выпадающий список с отображаемыми именами
            model_select = gr.Dropdown(
                choices=model_labels,
                label="Выберите модель",
                value=model_labels[0]  # первая метка, например "Pyannote (community-1)"
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
        inputs=[audio_input, min_sp, max_sp, model_select],
        outputs=[output_table, output_plot]
    )

    gr.Markdown(
        """
        **Примечания:**
        - **Pyannote**: Точная модель, требует GPU для лучшей производительности.
        - **DiariZen**: Быстрая модель, оптимизирована для CPU.
        - Аудио автоматически преобразуется в WAV, моно, 16 кГц.
        - Для русского языка качество может быть ниже.
        """
    )

if __name__ == "__main__":
    demo.launch(debug=True)