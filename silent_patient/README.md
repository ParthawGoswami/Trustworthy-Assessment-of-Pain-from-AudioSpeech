# Silent Patient — AI Pain Detector (TAME Pain Dataset)

Hackathon-ready demo that predicts pain from **speech or non-speech vocalizations** (sighs / breathing / groans), visualizes pain trends, and provides a **retrieval-only** (offline) “AI Nurse Assistant”.

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
- Model: small **CNN regressor/classifier** (fast, hackathon-friendly)
- Optional: a wav2vec2 fine-tuning script exists for later experimentation

## Retrieval-only Nurse Assistant (offline)

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
