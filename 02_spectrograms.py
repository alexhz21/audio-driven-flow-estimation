import os
import warnings

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Settings
PROJECT_DIR = r"C:\school\project\everything everything"
KEY_FILE = os.path.join(PROJECT_DIR, "everything_key.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "02_spectrogram_outputs")

AUDIO_COL = "audio_path"
TARGET_SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
OVERVIEW_COUNT = 12


def resolve_path(value):
    """Return an absolute project path."""
    path = str(value).strip()
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


def safe_name(text):
    """Create a valid file name."""
    invalid = '<>:"/\\|?*'
    return "".join("_" if char in invalid else char for char in text)


def calculate_spectrogram(audio_path):
    """Load audio and calculate a mel spectrogram."""
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    if len(y) == 0:
        raise ValueError("Empty audio file")

    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    return mel_db, sr, float(len(y) / sr)


def save_spectrogram(mel_db, sr, title, output_path):
    """Save one full spectrogram."""
    plt.figure(figsize=(10, 4.5))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
    )
    plt.colorbar(format="%+2.0f dB", label="Energy (dB)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def show_overview(items):
    """Display a sample of the saved results."""
    if not items:
        return

    shown = items[:OVERVIEW_COUNT]
    columns = 3
    rows = int(np.ceil(len(shown) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    image = None
    for axis, item in zip(axes, shown):
        image = librosa.display.specshow(
            item["mel_db"],
            sr=item["sr"],
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=axis,
        )
        axis.set_title(item["title"], fontsize=9)

    for axis in axes[len(shown):]:
        axis.axis("off")

    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), format="%+2.0f dB", shrink=0.75)

    fig.suptitle(f"Spectrogram overview: first {len(shown)} recordings", fontsize=15)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.90, hspace=0.45)

    overview_path = os.path.join(OUTPUT_DIR, "spectrogram_overview.png")
    plt.savefig(overview_path, dpi=220)
    plt.show()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== SPECTROGRAM GENERATOR ===")
    print(f"Reading: {KEY_FILE}")

    key = pd.read_excel(KEY_FILE)
    if AUDIO_COL not in key.columns:
        raise ValueError(f"Excel must contain '{AUDIO_COL}'")

    results = []
    overview_items = []
    total = len(key)

    for index, row in key.iterrows():
        audio_path = resolve_path(row[AUDIO_COL])
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        title = f"{index + 1}. {base_name}"
        output_name = f"{index + 1:03d}_{safe_name(base_name)}.png"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(audio_path)

            mel_db, sr, duration = calculate_spectrogram(audio_path)
            save_spectrogram(mel_db, sr, title, output_path)

            if len(overview_items) < OVERVIEW_COUNT:
                overview_items.append(
                    {"mel_db": mel_db, "sr": sr, "title": title}
                )

            results.append(
                {
                    "excel_row": index + 2,
                    "audio_path": audio_path,
                    "duration_s": duration,
                    "spectrogram_path": output_path,
                    "status": "ok",
                }
            )
            print(f"[{index + 1}/{total}] Saved | {output_name}")

        except Exception as error:
            results.append(
                {
                    "excel_row": index + 2,
                    "audio_path": audio_path,
                    "status": f"error: {error}",
                }
            )
            print(f"[{index + 1}/{total}] ERROR | {error}")

    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_DIR, "spectrogram_results.xlsx")
    results_df.to_excel(results_path, index=False)

    show_overview(overview_items)

    successful = int((results_df["status"] == "ok").sum())
    failed = len(results_df) - successful

    print("\n=== RESULTS ===")
    print(f"Total recordings: {len(results_df)}")
    print(f"Created:          {successful}")
    print(f"Failed:           {failed}")
    print(f"Sampling rate:    {TARGET_SR} Hz")
    print(f"Mel bands:        {N_MELS}")

    print("\n=== SAVED FILES ===")
    print(f"Spectrograms: {OUTPUT_DIR}")
    print(f"Status table: {results_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
