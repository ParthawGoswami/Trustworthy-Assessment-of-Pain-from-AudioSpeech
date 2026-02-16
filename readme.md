# TAME Pain Dataset

The TAME Pain Dataset consists of 7,039 annotated utterances totaling 311.24 minutes of audio data. Each utterance is labeled with self-reported pain levels and categorized based on the presence or absence of pain, pain severity, and experimental conditions (cold/warm). Additionally, the dataset includes annotations regarding audio quality and disturbances to assist in data preprocessing and analysis.

## Dataset Structure

The dataset is organized into the following main components:

1. **Audio Recordings**
2. **Metadata Files**
3. **Annotations**

## Accessing the Data

The TAME Pain Dataset is available on the [PhysioNet](https://physionet.org/) data platform. 

## File Descriptions

### Audio Recordings

- **File:** `mic1_trim_v1.zip`
- **Description:** Contains 51 subfolders, each representing a participant identified by a unique Participant ID (PID). Each subfolder includes audio recordings in `.wav` format, trimmed using Voice Activity Detection (VAD).
- **File Naming Convention:** `PID.COND.UTTNUM.UTTID.wav`
  - **PID:** Participant Identification (e.g., `p12345`)
  - **COND:** Experimental Condition (`LC`, `LW`, `RC`, `RW`)
  - **UTTNUM:** Utterance Number (sequential per condition)
  - **UTTID:** Utterance ID (corresponds to assigned sentence or `99999` for pain statements)

### Metadata Files

1. **Audio Metadata**
   - **File:** `meta_audio.csv`
   - **Description:** Contains metadata for each audio file.
   - **Columns:**
     - `PID`: Participant Identification
     - `COND`: Experimental Condition
     - `UTTNUM`: Utterance Number
     - `UTTID`: Utterance ID
     - `PAIN LEVEL`: Raw self-reported pain level
     - `REVISED PAIN`: Modified pain level aligned with the 1-10 scale
     - `DURATION`: Duration of the audio file in seconds
     - `ACTION LABEL`: Quality rating of the audio (0-4)
     - `NOTES`: Manual annotations and comments

2. **Participant Data**
   - **File:** `meta_participant.csv`
   - **Description:** Contains demographic and experimental data for each participant.
   - **Columns:**
     - `PID`: Participant Identification
     - `GENDER`: Self-reported gender
     - `AGE`: Age in years
     - `RACE/ETHNICITY`: Self-reported race/ethnicity
     - `FOLDER SIZE`: Storage size of audio files in megabytes
     - `NUMBER OF FILES`: Total number of audio files
     - `TOTAL DURATION`: Total duration of audio files in seconds
     - `LC`, `LW`, `RC`, `RW`: Completion status of each experimental condition (`1` for completed, `0` for incomplete)

### Annotations

- **Folder:** `Annotations`
- **Description:** Contains seven CSV files, each representing a distinct annotation category. Utterances can belong to multiple categories.

#### Annotation Files

1. **External Disturbances**
   - **File:** `External_Disturbances.csv`
   - **Description:** Records external noises unrelated to participant vocalization.
   - **Additional Column:** `NOISE RELATION` (Foreground, Background, or Both)

2. **Speech Errors and Disturbances**
   - **File:** `Speech_Errors_and_Disturbances.csv`
   - **Description:** Captures speech errors and verbal disturbances.

3. **Audio Cut Out**
   - **File:** `Audio_Cut_Out.csv`
   - **Description:** Notes instances where audio was cut, leading to loss of parts of sentences.

4. **Audible Breath**
   - **File:** `Audible_Breath.csv`
   - **Description:** Identifies audible inhales/exhales by participants.

5. **No Pain Rating So Copied**
   - **File:** `No_Pain_Rating_So_Copied.csv`
   - **Description:** Indicates audio files without a pain rating that were assigned a pain level based on adjacent ratings.

6. **No Assigned Sentence**
   - **File:** `No_Assigned_Sentence.csv`
   - **Description:** Marks audio files where the assigned sentence was not spoken.

7. **No Pain Rating**
   - **File:** `No_Pain_Rating.csv`
   - **Description:** Lists audio files without any pain rating and no adjacent ratings to copy from.

## Usage Instructions

1. **Download the Dataset:**
   - Access the dataset via the provided PhysioNet link.
   - Download the `mic1_trim_v1.zip` file and the associated metadata and annotation files.

2. **Extract Audio Recordings:**
   - Unzip `mic1_trim_v1.zip` to access participant folders containing `.wav` audio files.

3. **Understand File Naming:**
   - Use the file naming convention (`PID.COND.UTTNUM.UTTID.wav`) to navigate and identify specific audio files.

4. **Refer to Metadata:**
   - Utilize `meta_audio.csv` for detailed information about each audio file.
   - Use `meta_participant.csv` to access participant demographics and condition completion statuses.

5. **Handle Annotations:**
   - Explore the `Annotations` folder to understand and utilize the various annotation categories.
   - Multiple annotations for a single file are separated by semicolons in the `NOTES` column.

6. **Data Analysis:**
   - Leverage the `ACTION LABEL` for filtering audio quality.
   - Use pain level annotations for classification tasks (Binary, Three-Class, Condition).

## Additional Details

- **Pain Level Adjustments:**
  - Original pain levels of `0` were relabeled to `1` to maintain a 1-10 scale.
  - Pain levels are categorized into:
    - **Binary Task:** No Pain (1-3), Pain (4-10)
    - **Three-Class Task:** Mild (1-3), Moderate (4-6), Severe (7-10)
    - **Condition Task:** Warm (`LW`, `RW`), Cold (`LC`, `RC`)

- **Audio Quality Labels:**
  - `ACTION LABEL` ranges from `0` (highest quality) to `4` (lowest quality), based on the presence of disturbances and errors.

- **Background Noise:**
  - All recordings have a uniform background fan noise, with varying intensities across participants.

- **Exclusion of Unlabeled Utterances:**
  - Five utterances without pain ratings were excluded from the dataset.


# Silent Patient — AI Pain Detector (TAME Pain Dataset)

Hackathon-ready demo that predicts pain from **speech or non-speech vocalizations** (sighs / breathing / groans), visualizes pain trends, and provides a **retrieval-only** “AI Nurse Assistant”.

What it supports:

- **Pain regression**: predict pain `1–10` based on `REVISED PAIN`
- **Pain classification**: mild / moderate / severe (optional training task)
- **Telemedicine mode**: **live pain trend** chart that updates as you record/upload clips
- **Dataset mode**: **patient-specific pain trend** from `meta_audio.csv` with a **PID dropdown**
- **No-speech mode (heuristic)**: flags breathing/gasp-like audio when speech isn’t present
- **Detected cues (heuristic acoustic features)**: breathiness / pitch instability / reduced speech rate signals for judge-friendly explainability
- **AI Nurse Assistant (retrieval-only, offline)**:
  - Retrieves from a local markdown knowledge base (no LLM calls)
  - Personalizes guidance by patient type (adult / post-op / pediatric)
  - Uses the **dataset-selected patient trend** analysis to drive triage framing
  - Shows concise summaries by default; full sources are hidden under an **audit** expander

> Important: The dataset/license is PhysioNet Restricted Health Data License; do not redistribute audio.

## Project layout

- `silent_patient/src/` — Python package
- `silent_patient/app/streamlit_app.py` — Streamlit demo UI (live + dataset trend + AI Nurse Assistant)
- `silent_patient/src/silent_patient/scripts/train.py` — training (writes `model_bundle.pt` for the UI)
- `silent_patient/src/silent_patient/rag.py` — offline retriever (`LocalRagIndex`) + query builder
- `silent_patient/kb/pain_protocols.md` — offline knowledge base (markdown)

## Quickstart (macOS)

### 1) Extract audio zip

From dataset root:

```zsh
mkdir -p .cache/audio
unzip -q mic1_trim_v1 -d .cache/audio
```

This creates `.cache/audio/mic1_trim_v2/.../*.wav`.

### 2) Install

Use any env manager you like. With pip:

```zsh
python -m pip install -e ".[dev]"
```

### 3) Train baseline model

This trains a small CNN baseline and writes a Streamlit-ready bundle at:
`.cache/models/baseline/model_bundle.pt`.

```zsh
python -m silent_patient.scripts.train \
  --meta-csv meta_audio.csv \
  --audio-root .cache/audio/mic1_trim_v2 \
  --outdir .cache/models/baseline \
  --task regression
```

### 4) Run Streamlit demo

The UI defaults to:

- Model bundle: `.cache/models/baseline/model_bundle.pt`
- Knowledge base: `silent_patient/kb/`
- Dataset metadata for trends: `meta_audio.csv`

```zsh
streamlit run silent_patient/app/streamlit_app.py
```

## What the demo shows

- Record audio in-browser or upload a clip
- Predict:
  - **Pain level** (1–10)
  - **Confidence** (hackathon-friendly heuristic)
  - **Detected cues** (simple acoustic feature heuristics)
- Track:
  - **Live session trend** (what you record during the session)
  - **Dataset trend per patient (PID)** using `meta_audio.csv`
- Ask:
  - **AI Nurse Assistant (retrieval-only)** for protocol-aligned next steps + explainability
  - The assistant incorporates the **dataset-selected PID trend** (n/min/max/mean/last + trend note) when available

## Model choices (baseline)

- Feature extraction: log-mel spectrogram + MFCC summary stats
- Model: small **CNN regressor/classifier** (fast)
- Optional: a wav2vec2 fine-tuning script exists for later experimentation

## Retrieval-only Nurse Assistant

The assistant is intentionally **not** a chat LLM. It uses local retrieval only:

- Index: `silent_patient/src/silent_patient/rag.py` (`LocalRagIndex`)
- Knowledge base: `silent_patient/kb/*.md`

In the Streamlit UI:

- Main assistant output is a **short summary**.
- Retrieved protocol chunks are available under **Show retrieved sources (audit)**.

## Notes / troubleshooting

- If the dataset trend sidebar says `meta_audio.csv not found`, confirm you’re running Streamlit from the **dataset root**.
- If the model bundle isn’t found, train once (Step 3) or update the sidebar path.
- If you see Streamlit warnings about `use_container_width`, they’re deprecation warnings and not a functional error.

## Next steps

- Split the knowledge base by patient type (adult / post-op / peds) to reduce cross-retrieval
- Expand cue heuristics to better separate breathiness vs. noise/artifacts
- Add stronger alert thresholds (vitals/context) and demo a clear “why this alert” trace
- Improve model quality with the included wav2vec2 fine-tuning pipeline
