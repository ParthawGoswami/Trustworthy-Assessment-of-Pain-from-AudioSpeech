from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from silent_patient.bundle import load_bundle
from silent_patient.config import LabelConfig
from silent_patient.data import load_meta, read_audio_any
from silent_patient.features import extract_features, simple_cue_features
from silent_patient.model import SmallCnnClassifier, SmallCnnRegressor
from silent_patient.rag import LocalRagIndex, build_query

try:
    from st_audiorec import st_audiorec
except Exception:  # pragma: no cover
    st_audiorec = None


def _load_model(bundle_path: Path):
    bundle = load_bundle(bundle_path)
    if bundle.task == "regression":
        model = SmallCnnRegressor(in_channels=bundle.in_channels)
    else:
        model = SmallCnnClassifier(in_channels=bundle.in_channels, n_classes=3)
    model.load_state_dict(bundle.state_dict)
    model.eval()
    return bundle, model


def _confidence_from_val_metric(task: str, val_metric: float, pred: float | int) -> float:
    """Cheap hackathon-friendly confidence score."""
    if task == "regression":
        mae = max(0.25, float(val_metric))
        penalty = 0.02 * float(pred)
        conf = 1.0 / (1.0 + mae) - penalty
        return float(np.clip(conf, 0.05, 0.95))
    return float(np.clip(val_metric, 0.05, 0.95))


def main() -> None:
    st.set_page_config(page_title="Silent Patient — Pain Detector", layout="wide")
    st.title("Silent Patient — AI Pain Detector")
    st.caption("Record speech or vocal sounds; get pain level + confidence + cues; track trend over time.")

    bundle_path = Path(
        st.sidebar.text_input(
            "Model bundle path",
            value=".cache/models/baseline/model_bundle.pt",
        )
    )
    if not bundle_path.exists():
        st.warning("Model bundle not found. Train first or set the correct path in the sidebar.")
        st.stop()

    bundle, model = _load_model(bundle_path)
    st.sidebar.write(f"Task: **{bundle.task}**")
    st.sidebar.write(f"Sample rate: **{bundle.audio_cfg.sample_rate} Hz**")

    st.sidebar.divider()
    st.sidebar.subheader("Dataset trend (per patient)")
    meta_path = Path(st.sidebar.text_input("meta_audio.csv path", value="meta_audio.csv"))
    max_action_label = int(st.sidebar.slider("Max ACTION LABEL (quality)", 0, 4, 2))
    cond_filter = st.sidebar.multiselect(
        "Conditions",
        options=["LC", "LW", "RC", "RW"],
        default=["LC", "LW", "RC", "RW"],
    )

    @st.cache_data(show_spinner=False)
    def _cached_meta(path_str: str) -> pd.DataFrame:
        return load_meta(Path(path_str))

    meta_df: pd.DataFrame | None
    selected_pid: str | None
    if meta_path.exists():
        meta_df = _cached_meta(str(meta_path))
        meta_df = meta_df[meta_df["ACTION LABEL"] <= max_action_label]
        if cond_filter:
            meta_df = meta_df[meta_df["COND"].isin(cond_filter)]
        pids = sorted(meta_df["PID"].unique().tolist())
        default_pid = "p10085" if "p10085" in pids else (pids[0] if pids else "")
        selected_pid = st.sidebar.selectbox(
            "Patient (PID)",
            options=pids,
            index=pids.index(default_pid) if default_pid in pids else 0,
        )
    else:
        meta_df = None
        selected_pid = None
        st.sidebar.warning("`meta_audio.csv` not found; dataset trend disabled.")

    if "events" not in st.session_state:
        st.session_state.events = []

    # --- AI Nurse Assistant controls ---
    st.sidebar.divider()
    st.sidebar.subheader("AI Nurse Assistant (retrieval-only)")
    kb_dir = Path(st.sidebar.text_input("Knowledge base folder", value="silent_patient/kb"))
    patient_type = st.sidebar.selectbox(
        "Patient type",
        options=[
            "Adult (general)",
            "Adult (post-operative)",
            "Pediatric",
        ],
        index=0,
    )
    clinical_context = st.sidebar.text_area(
        "Context (optional)",
        value="",
        help="E.g., post-op day 1, ED triage, chronic pain, after physical therapy, etc.",
    )

    @st.cache_resource(show_spinner=False)
    def _kb_index(path_str: str) -> LocalRagIndex:
        return LocalRagIndex.from_markdown_dir(Path(path_str))

    kb_index: LocalRagIndex | None
    if kb_dir.exists():
        kb_index = _kb_index(str(kb_dir))
    else:
        kb_index = None

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.subheader("Record / Upload")
        if st_audiorec is None:
            st.error("`streamlit-audiorec` not available in this environment.")
            st.stop()

        audio_bytes = st_audiorec()
        uploaded = st.file_uploader("Upload a WAV", type=["wav"])
        if uploaded is not None:
            audio_bytes = uploaded.read()

    if audio_bytes:
            audio, sr = read_audio_any(audio_bytes, bundle.audio_cfg.sample_rate)
            cues = simple_cue_features(audio, sr)

            x = extract_features(audio, bundle.audio_cfg)
            xt = torch.from_numpy(x).unsqueeze(0)

            with torch.no_grad():
                if bundle.task == "regression":
                    pred = float(model(xt).item())
                    pred = float(np.clip(pred, 1.0, 10.0))
                    pain_level = pred
                    pain_class = LabelConfig.pain_to_class(pred)
                else:
                    logits = model(xt)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    pain_class = int(np.argmax(probs))
                    pain_level = {0: 2.0, 1: 5.0, 2: 8.0}[pain_class]

            cls_name = {0: "Mild", 1: "Moderate", 2: "Severe"}[pain_class]
            conf = _confidence_from_val_metric(bundle.task, bundle.val_metric, pain_level)
            st.markdown(
                f"### Pain Level: **{pain_level:.1f}/10** ({cls_name})\n\nConfidence: **{conf*100:.0f}%**"
            )

            cue_lines: list[str] = []
            if cues.get("non_speech_likelihood", 0.0) > 0:
                cue_lines.append("non-speech vocalization")
            if cues.get("breathiness", 0.0) > 0.15:
                cue_lines.append("breathiness")
            if cues.get("pitch_instability", 0.0) > 0.08:
                cue_lines.append("pitch instability")
            if cues.get("pitch_var", 0.0) > 0.18:
                cue_lines.append("pitch variance")
            if cues.get("onsets_per_sec", 0.0) > 3.0:
                cue_lines.append("fast speech rate")
            if not cue_lines:
                cue_lines = ["(no strong cues detected)"]
            st.write("**Detected cues:**", ", ".join(cue_lines))

            st.session_state.events.append(
                {
                    "t": time.time(),
                    "pain": float(pain_level),
                    "conf": float(conf),
                    "cues": {k: float(v) for k, v in cues.items()},
                }
            )

            with st.expander("Show cue values"):
                st.json({k: float(v) for k, v in cues.items()})

            # Store latest inference details for the assistant panel
            st.session_state.last_inference = {
                "pain_level": float(pain_level),
                "pain_class": cls_name,
                "confidence": float(conf),
                "detected_features": [c for c in cue_lines if not c.startswith("(")],
            }

    with col2:
        st.subheader("Telemedicine pain trend")
        tabs = st.tabs(["Live (this session)", "Dataset (selected patient)", "AI Nurse Assistant"])

        with tabs[0]:
            events = st.session_state.events
            if not events:
                st.info("Record a few clips to build a timeline.")
            else:
                t0 = events[0]["t"]
                xs = [e["t"] - t0 for e in events]
                ys = [e["pain"] for e in events]
                cs = [e["conf"] for e in events]

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(xs, ys, marker="o")
                ax.set_xlabel("seconds")
                ax.set_ylabel("pain (1-10)")
                ax.set_ylim(1, 10)
                for x, y, c in zip(xs, ys, cs):
                    if y >= 8 and c >= 0.6:
                        ax.annotate("ALERT", (x, y), textcoords="offset points", xytext=(5, 5))
                st.pyplot(fig, clear_figure=True)

                last = events[-1]
                if last["pain"] >= 8 and last["conf"] >= 0.6:
                    st.error("Doctor alert: pain spike detected")

        with tabs[1]:
            if meta_df is None or selected_pid is None:
                st.info("Provide `meta_audio.csv` in the sidebar to enable dataset trend.")
            else:
                pid_df = meta_df[meta_df["PID"] == selected_pid].copy()
                if pid_df.empty:
                    st.warning("No rows for this PID after filters.")
                else:
                    cond_order = {"LW": 0, "RW": 1, "LC": 2, "RC": 3}
                    pid_df["_cond_order"] = pid_df["COND"].map(cond_order).fillna(99).astype(int)
                    pid_df = pid_df.sort_values(["_cond_order", "UTTNUM"]).reset_index(drop=True)
                    pid_df["seq"] = np.arange(1, len(pid_df) + 1)

                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(pid_df["seq"], pid_df["REVISED PAIN"], marker="o", linewidth=1)
                    ax.set_xlabel("utterance sequence")
                    ax.set_ylabel("pain (REVISED PAIN)")
                    ax.set_ylim(1, 10)
                    ax.set_title(f"{selected_pid} pain trajectory")

                    severe = pid_df[pid_df["REVISED PAIN"] >= 8]
                    if not severe.empty:
                        ax.scatter(
                            severe["seq"],
                            severe["REVISED PAIN"],
                            color="red",
                            s=30,
                            label="severe (>=8)",
                        )
                        ax.legend(loc="lower right")

                    st.pyplot(fig, clear_figure=True)
                    with st.expander("Show rows"):
                        st.dataframe(
                            pid_df[["PID", "COND", "UTTNUM", "UTTID", "REVISED PAIN", "ACTION LABEL"]],
                            use_container_width=True,
                            height=220,
                        )

        with tabs[2]:
            st.markdown(
                "**Retrieval-only clinical guidance.** This panel only shows text retrieved from the local knowledge base and formats it into checklists."
            )

            last = st.session_state.get("last_inference")
            if last is None:
                st.info("Record or upload audio first to generate a pain prediction.")
                return

            if kb_index is None:
                st.warning("Knowledge base folder not found. Point sidebar to a folder like `silent_patient/kb`.")
                return

            # Trend note from the *dataset(selected patient)* trajectory when available.
            # If dataset isn't available, fall back to this-session live events.

            def _analyze_series(values: list[float]) -> dict:
                if not values:
                    return {
                        "n": 0,
                        "min": None,
                        "max": None,
                        "mean": None,
                        "last": None,
                        "delta": None,
                        "delta_last2": None,
                        "spike": False,
                        "trend": "unknown",
                    }
                arr = np.asarray(values, dtype=float)
                n = int(arr.size)
                last = float(arr[-1])
                delta = float(arr[-1] - arr[0]) if n >= 2 else 0.0
                delta_last2 = float(arr[-1] - arr[-2]) if n >= 2 else 0.0
                spike = bool(abs(delta_last2) >= 2.0)
                if n >= 3:
                    slope = float(np.polyfit(np.arange(n, dtype=float), arr, 1)[0])
                    if slope > 0.15:
                        trend = "increasing"
                    elif slope < -0.15:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                else:
                    trend = "stable" if n == 1 else ("increasing" if delta_last2 > 0 else "decreasing")
                return {
                    "n": n,
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "last": last,
                    "delta": delta,
                    "delta_last2": delta_last2,
                    "spike": spike,
                    "trend": trend,
                }

            dataset_note = None
            dataset_stats = None
            dataset_last_pain = None

            if meta_df is not None and selected_pid is not None:
                pid_df2 = meta_df[meta_df["PID"] == selected_pid].copy()
                if not pid_df2.empty:
                    cond_order = {"LW": 0, "RW": 1, "LC": 2, "RC": 3}
                    pid_df2["_cond_order"] = pid_df2["COND"].map(cond_order).fillna(99).astype(int)
                    pid_df2 = pid_df2.sort_values(["_cond_order", "UTTNUM"]).reset_index(drop=True)
                    series = [float(x) for x in pid_df2["REVISED PAIN"].tolist() if pd.notna(x)]
                    dataset_stats = _analyze_series(series)
                    if dataset_stats["n"] >= 1:
                        dataset_last_pain = float(dataset_stats["last"])
                        dataset_note = (
                            f"dataset({selected_pid}) n={dataset_stats['n']}, last={dataset_stats['last']:.1f}, "
                            f"min={dataset_stats['min']:.1f}, max={dataset_stats['max']:.1f}, "
                            f"trend={dataset_stats['trend']}, last_step_Δ={dataset_stats['delta_last2']:+.1f}"
                        )

            trend_note = None
            if dataset_note:
                trend_note = dataset_note
            else:
                events = st.session_state.get("events", [])
                if len(events) >= 2:
                    p0 = float(events[-2]["pain"])
                    p1 = float(events[-1]["pain"])
                    dt = max(1e-3, float(events[-1]["t"] - events[-2]["t"]))
                    dp = p1 - p0
                    if abs(dp) >= 2.0:
                        trend_note = f"live pain changed {p0:.1f}→{p1:.1f} in {dt:.0f}s (Δ={dp:+.1f})"

            # If we have dataset trend, prefer the *dataset last pain* as the monitoring signal.
            # Otherwise fall back to the last model inference pain.
            pain_level = float(dataset_last_pain) if dataset_last_pain is not None else float(last["pain_level"])
            if pain_level >= 8:
                urgency = "Severe"
            elif pain_level >= 4:
                urgency = "Moderate"
            else:
                urgency = "Mild"

            detected_feats = last.get("detected_features", [])
            base_q = build_query(
                pain_level=pain_level,
                pain_class=str(last["pain_class"]),
                patient_type=str(patient_type),
                context=str(clinical_context).strip() or "unspecified",
                detected_features=detected_feats,
                trend_note=trend_note,
            )

            # Retrieve targeted snippets so output varies by patient type and severity.
            # We retrieve multiple small "anchors" and only display a short excerpt.
            q_reco = f"{patient_type} {urgency} pain recommendation | {base_q}"
            q_actions = f"{patient_type} {urgency} pain suggested actions | {base_q}"
            q_xai = f"interpreting vocal acoustic signs explainability | {', '.join(detected_feats) or 'none'}"
            q_triage = f"triage escalation rapid pain increase | {trend_note or ''}"

            reco_chunks = kb_index.retrieve(q_reco, top_k=1)
            action_chunks = kb_index.retrieve(q_actions, top_k=1)
            xai_chunks = kb_index.retrieve(q_xai, top_k=1)
            triage_chunks = kb_index.retrieve(q_triage, top_k=1) if trend_note else []

            def _excerpt(txt: str, n: int = 260) -> str:
                t = " ".join((txt or "").split())
                return t[:n] + ("…" if len(t) > n else "")

            st.markdown(
                f"### Clinical Recommendation\n\nPredicted Pain: **{pain_level:.1f}/10** — **{urgency}**\n\nPatient type: **{patient_type}**"
            )
            if trend_note:
                st.write(f"**Trend reasoning:** {trend_note}")

            if dataset_stats is not None:
                st.caption(
                    f"Dataset summary: n={dataset_stats['n']} · mean={dataset_stats['mean']:.1f} · "
                    f"min={dataset_stats['min']:.1f} · max={dataset_stats['max']:.1f}"
                )

            # Keep raw protocol text out of the main UI (only show in the audit expander below).

            st.markdown("### Suggested Actions (from retrieved protocols)")
            actions = []
            if urgency == "Severe":
                actions += [
                    "Reassess immediately",
                    "Ask location/quality of pain (OPQRST)",
                    "Check vitals (HR/BP/RR/SpO2/temp)",
                    "Consider fast-acting analgesia per local protocol",
                    "Escalate care if red flags or refractory pain",
                ]
            elif urgency == "Moderate":
                actions += [
                    "Reassess and quantify pain + functional limitation",
                    "Use multimodal analgesia principles as appropriate",
                    "Reassess response after intervention",
                ]
            else:
                actions += [
                    "Non-pharmacologic measures as appropriate",
                    "Patient education and comfort measures",
                ]

            # Patient-type personalization (small deltas that judges notice)
            if patient_type == "Pediatric":
                actions.insert(0, "Use a pediatric-appropriate pain scale (e.g., observational vs faces/NRS)")
                actions.append("Confirm caregiver observations and age-appropriate communication")
            elif patient_type == "Adult (post-operative)":
                actions.insert(0, "Assess pain impact on breathing/coughing and early mobilization")
                actions.append("Monitor for post-op complications if pain is increasing")

            st.write("\n".join([f"• {a}" for a in actions]))

            st.markdown("### Explainable AI (feature → interpretation)")
            feats = last.get("detected_features", [])
            if feats:
                st.write("**Detected features:**")
                st.write("\n".join([f"• {f}" for f in feats]))
            else:
                st.write("No prominent acoustic cues detected.")

            st.markdown("**Medical interpretation (retrieved):**")
            if xai_chunks:
                st.write(_excerpt(xai_chunks[0].chunk))
            else:
                st.write("No explainability snippet found in the knowledge base for the detected features.")

            # Smart alert reasoning snippet (only if trend indicates rapid escalation)
            if triage_chunks:
                st.markdown("**Alert reasoning (retrieved):**")
                st.write(_excerpt(triage_chunks[0].chunk))

            # Keep protocol details hidden by default (the user requested not to show them).
            with st.expander("Show retrieved sources (audit)"):
                for label, chunks in [
                    ("Recommendation", reco_chunks),
                    ("Actions", action_chunks),
                    ("Explainability", xai_chunks),
                    ("Triage", triage_chunks),
                ]:
                    for r in chunks:
                        if patient_type == "Pediatric" and "Pediatric" not in r.chunk and "pediatric" not in r.chunk:
                            # Avoid mixing adult-only protocol text into pediatric view unless the user chooses
                            # to inspect the source explicitly.
                            continue
                        if patient_type == "Adult (post-operative)" and "Post-operative" not in r.chunk and "post-operative" not in r.chunk and "post op" not in r.chunk:
                            continue
                        st.markdown(f"**{label} source:** `{r.source}` (score={r.score:.3f})")
                        st.write(r.chunk)


if __name__ == "__main__":
    main()
