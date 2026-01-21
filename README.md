# JASSQA: Joint Acoustic-Semantic Speech Quality Assessment

Official repository for the paper: **"Bridging the Semantic Gap: Cross-Attentive Fusion for Joint Acoustic-Semantic Speech Quality Assessment"** (ICASSP 2026).

JASSQA is a non-intrusive speech quality assessment (SQA) model featuring a dual-branch architecture. It integrates specialized **acoustic features** (via DAC Tokenizer) and **semantic representations** (via Whisper) through a bidirectional cross-attention mechanism to better align with human quality perception.

## 1. Repository Status

This repository currently contains the **Testing (Inference) Module** of JASSQA. You can use the provided scripts to evaluate speech quality on various datasets.

---

## 2. Data Preparation

To run the inference script, you need to prepare a metadata file in `.csv` format.

### CSV Requirements
Based on the provided `NISQA_TEST_P501_file.csv`, your input CSV should follow this structure:

* **Required Column**:
    * `filepath_deg`: The relative or absolute path to the degraded audio file (e.g., `.wav`).
* **Optional Columns (for evaluation metrics)**:
    * `mos`: Ground truth Mean Opinion Score to calculate PCC, SRCC, and RMSE.
    * `db`: Dataset name or identifier.

### Audio Requirements
* **Format**: Standard `.wav` files.
* **Preprocessing**: The model handles feature extraction using DAC and Whisper; ensure your audio files are accessible via the paths provided in the CSV.

---

## 3. Eval

### Environment Setup
Ensure you have the following dependencies installed:
* `PyTorch`
* `descript-audio-codec` (for DAC acoustic tokens)
* `openai-whisper` (for semantic features)
* `pandas`, `numpy`, `matplotlib`

### Running Inference
We provide a bash script `test_dac_whisper_JASSQA.sh` to automate the testing process.

1.  **Configure the script**:
    Open `test_dac_whisper_JASSQA.sh` and set your local paths:
    ```bash
    CHECKPOINT_PATH="/path/to/your/model_checkpoint.pt"
    TEST_DATASET="/path/to/your/metadata_file.csv"
    AUDIO_ROOT_DIR="/path/to/your/audio_folder"
    ```

2.  **Execute**:
    ```bash
    ./test_dac_whisper_JASSQA.sh
    ```

The script will output:
* `test_predictions.csv`: Contains the predicted scores for each audio file.
* `scatterplot.png`: A visualization of the correlation between predictions and ground truth (if MOS is provided).

---
