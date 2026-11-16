# Engine-Sound Vehicle Identification — Hierarchical Deep Learning

A PyTorch-based **Hierarchical Multi-Task Convolutional Neural Network (CNN)** that identifies vehicle profiles directly from raw engine audio recordings (`.m4a` / `.wav`). Instead of flat classification, the model uses a **5-head multi-task architecture** to predict foundational engine traits first, acting as a cascading filter to narrow down and rank candidate vehicles.

## Key Highlights & Results
* **Hierarchical 5-Head Architecture:** Simultaneously predicts core engine traits (cylinder count, fuel type, aspiration/turbo, engine family, and vehicle profile) to constrain and improve fine-grained vehicle ranking, achieving **~70% test accuracy**.
* **Custom Audio Signal Preprocessing:** Automated pipeline handling `.m4a` and `.wav` inputs via `ffmpeg`, mono resampling, bandpass filtering to isolate engine harmonics, steady-segment detection (separating idle vs. rev states), normalization, and log-Mel spectrogram extraction.
* **End-to-End Inference:** Includes both the full experimental training notebook (`car_pred.ipynb`) and a standalone Python inference script (`inference.py`) for single-file predictions.

## Tech Stack
* **Deep Learning:** Python, PyTorch, TorchAudio
* **Audio & Signal Processing:** `ffmpeg`, Librosa, SciPy, NumPy
* **Environment:** Jupyter Notebook, VS Code

## Repository Structure
* `notebook/car_pred.ipynb` — End-to-end training workflow, spectrogram visualizations, and validation metrics.
* `inference.py` — Standalone script to run predictions on an engine audio recording.
* `weights/master_5head_engine_model.pth` — Saved 5-head PyTorch model weights.
* `sampleaudio/alto.m4a` — Sample engine recording for quick inference testing.
* `requirements.txt` — Python package dependencies.

## Quick Start (Run Inference)

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com
   cd engine-sound-vehicle-identification
   pip install -r requirements.txt
   ```
   *(Note: Ensure `ffmpeg` is installed on your system to decode `.m4a` files.)*

2. **Run inference on the sample audio file:**
   ```bash
   python inference.py --audio sampleaudio/alto.m4a --weights weights/master_5head_engine_model.pth
   ```

## Pipeline Overview
1. **Ingestion & Cleaning:** Convert raw `.m4a`/`.wav` recordings to mono PCM audio and apply a bandpass filter to strip background wind and road noise.
2. **Segmentation & Feature Extraction:** Detect steady-state engine segments and compute log-Mel spectrograms.
3. **Hierarchical Prediction:** Pass spectrograms through shared convolutional feature extractors into 5 task-specific classification heads to predict foundational engine traits (cylinder count, fuel type, aspiration, and engine family) and rank matching vehicle profiles.

## Dataset Constraints & Engineering Trade-offs
* **Data Scarcity & Generalization:** Because cleanly labeled, multi-state (idle vs. rev) vehicle engine recordings are limited in volume, fine-grained vehicle classification faces natural generalization ceilings (~70% test accuracy).
* **Architectural Mitigation:** To prevent overfitting on a small dataset, the pipeline uses steady-state audio segmentation to extract multiple clean training windows per recording and leverages a **5-head hierarchical multi-task design** so shared engine traits regularize the fine-grained vehicle predictions.
* **Future Improvements:** Expanding the custom audio dataset across more acoustic environments and incorporating SpecAugment or pitch-shifting augmentations would further improve out-of-distribution accuracy.
