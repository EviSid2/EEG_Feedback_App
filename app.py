
import os
import io
import json
import glob
import time
import datetime as dt
from typing import Dict, Any, Optional
from requests.exceptions import ReadTimeout, ConnectionError
import random 
import re
import pandas as pd
import numpy as np
import numpy as _np
import pandas as _pd
import streamlit as st
import requests
import altair as alt

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b-instruct-q4_K_M"
MODEL_NAME = "qwen2.5:1.5b-instruct-q4_K_M" 
FAST_OLLAMA_MODEL = "qwen2.5:1.5b-instruct-q4_K_M"  
NUM_THREADS = int(os.environ.get("OLLAMA_NUM_THREAD", "8"))
os.environ.setdefault("OLLAMA_NUM_THREAD", "8")

def compute_stability_score(x: np.ndarray) -> float:
    """Higher = more stable (smoother) relative to typical amplitude."""
    x = np.asarray(x, dtype=float)
    if x.size < 5:
        return 0.0
    diffs = np.diff(x)
    mad_signal = np.median(np.abs(x - np.median(x))) + 1e-9
    mad_diffs  = np.median(np.abs(diffs - np.median(diffs))) + 1e-9
    raw = 1.0 - (mad_diffs / mad_signal)
    return float(np.clip(raw, 0.0, 1.0))

def calm_badge(score: float) -> str:
    """Simple text band for calm level."""
    if score < 0.2:
        return "stormy"
    elif score < 0.4:
        return "settling"
    elif score < 0.6:
        return "steadying"
    elif score < 0.8:
        return "smooth"
    else:
        return "flow"

def score_to_color(score: float) -> str:
    """Map 0..1 -> red→amber→teal for branding."""
    red   = (217, 83, 79)    # #D9534F
    amber = (240, 173, 78)   # #F0AD4E
    teal  = (74, 163, 162)   # #4AA3A2
    def lerp(a,b,t): return tuple(int(a[i] + (b[i]-a[i]) * max(0,min(1,t))) for i in range(3))
    if score <= 0.5:
        rgb = lerp(red, amber, (score/0.5))
    else:
        rgb = lerp(amber, teal, (score-0.5)/0.5)
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

def focus_band(score: float) -> str:
    """Label for focus (attention) strength."""
    if score < 0.2:
        return "scattered"
    elif score < 0.4:
        return "drifting"
    elif score < 0.6:
        return "balancing"
    elif score < 0.8:
        return "engaged"
    else:
        return "laser-like"               
        
def ollama_ready(url: str = "http://localhost:11434/api/tags", timeout: int = 2) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def prewarm_ollama(model: str = FAST_OLLAMA_MODEL):
    """Load the model into memory once so the first real call is fast."""
    if not ollama_ready():
        return
    if st.session_state.get("ollama_warmed"):
        return
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": "Say OK.",
                "stream": False,
                "keep_alive": "2h",
                "options": {
                    "num_predict": 5,
                    "temperature": 0.1,
                    "num_thread": int(os.environ.get("OLLAMA_NUM_THREAD", "8")),
                },
            },
            timeout=10,
        )
        st.session_state["ollama_warmed"] = True
    except Exception:
        pass

def call_ollama(prompt: str, temperature: float = 0.6, timeout: int = 240) -> str:
    """Send a prompt to Ollama and return the model's text response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Couldn't reach Ollama at {OLLAMA_URL}. Is it running? Error: {e}"
        )

def call_ollama_stream_buffered(prompt: str,
                                            model: str = FAST_OLLAMA_MODEL,
                                            temperature: float = 0.4,
                                            num_predict: int = 420,  
                                            timeout: int = 180):
            
                url = "http://localhost:11434/api/generate"
                payload = {
                    "model": model,
                    "stream": True,
                    "keep_alive": "30m",
                    "prompt": prompt,
                    "options": {
                        "temperature": temperature,
                        "num_predict": num_predict,
                        "num_thread": NUM_THREADS,
                    }
                }
                with requests.post(url, json=payload, timeout=timeout, stream=True) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            continue
                        yield chunk.get("response", "")       

DEBUG = False

def dev_show(x):
    if DEBUG:
        st.json(x)

# --- Initialize current page ---
if "page" not in st.session_state:
    st.session_state["page"] = "Start"

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="What's The Verdict?",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCH_DEFAULT = os.path.join(BASE_DIR, "OpenBCI_Exports", "Recordings")
RATINGS_FILE = os.path.abspath("./ratings.csv")

def ai_coach_quick_reflection(score=None, participant=None, task=None, profile=None):
    """
    Short AI insight for Visual / Literal modes (2–3 sentences).
    Uses the full enhanced_summary (calm, focus, movement, bands, duration)
    plus optional personal profile. The old `score` argument is kept only
    as a fallback; the main signal comes from st.session_state["summary"].
    """

    # --- pull metrics from summary ---
    summary = st.session_state.get("summary") or {}
    calm_val     = summary.get("calm_score")        
    focus_val    = summary.get("focus_score")      
    movement_val = summary.get("movement_score")   
    movement_lvl = summary.get("movement_level") or summary.get("movement")
    dom_band     = summary.get("dominant_band")
    duration_sec = summary.get("duration_sec")
    task_label   = summary.get("task_label") or task or "your session"

    # If we still have nothing, fall back to the raw score that was passed in
    if calm_val is None and score is not None:
        try:
            calm_val = float(score) * 100.0 if 0.0 <= score <= 1.0 else float(score)
        except Exception:
            calm_val = None

    # --- build a compact, factual description for the model ---
    fact_lines = []

    if calm_val is not None:
        fact_lines.append(
            f"- Calm / stability score: about {calm_val:.0f}/100 "
            f"(higher = smoother, more even brainwaves)."
        )
    if focus_val is not None:
        fact_lines.append(
            f"- Focus score: about {focus_val:.0f}/100 "
            f"(higher = more sustained, task-aligned attention)."
        )
    if movement_val is not None:
        fact_lines.append(
            f"- Movement stillness: about {movement_val:.0f}/100 "
            f"(higher = less head and facial movement in the signal)."
        )
    if movement_lvl is not None:
        fact_lines.append(f"- Overall movement level: {movement_lvl}.")
    if dom_band:
        fact_lines.append(
            f"- Dominant frequency band: {dom_band} "
            f"(rough indicator of whether the brain was more relaxed, sleepy, or alert)."
        )
    if duration_sec is not None:
        mins = int(round(duration_sec / 60.0))
        fact_lines.append(f"- Session length: about {mins} minute(s).")

    if not fact_lines:
        fact_lines.append(
            "- No detailed metrics were available; treat this as a very high-level impression only."
        )

    metrics_block = "\n".join(fact_lines)

    # --- personal context (only if consented) ---
    personal_bits = []
    if isinstance(profile, dict) and profile.get("consent"):
        if profile.get("primary_goal"):
            personal_bits.append(f"Goal: {profile['primary_goal']}.")
        if profile.get("eeg_experience"):
            personal_bits.append(f"EEG experience: {profile['eeg_experience']}.")
        if profile.get("age_range"):
            personal_bits.append(f"Age range: {profile['age_range']}.")
        if profile.get("gender"):
            personal_bits.append(f"Gender: {profile['gender']}.")
        if profile.get("occupation"):
            personal_bits.append(f"Occupation: {profile['occupation']}.")
    personal_str = " ".join(personal_bits) if personal_bits else "not provided."

    # --- system instructions: ---
    SYSTEM_PROMPT = (
        "You are a calm, encouraging mindfulness and focus coach. "
        "You see EEG-based scores for calm/stability, focus, movement, and dominant band. "
        "Give feedback that matches these scores honestly: if calm or focus are low, "
        "say clearly that the mind looked restless or changeable; do NOT only describe the session as calm. "
        "If scores are high, you can highlight that as a strength. "
        "Use everyday language like 'your brainwaves jumped around a lot' or "
        "'they stayed smooth for most of the session', and avoid technical terms. "
        "Keep the response very short: 2 or 3 sentences maximum."
    )

    USER_PROMPT = f"""
    Participant ID: {participant or "Participant"}
    Task: {task_label}

    EEG metrics:
    {metrics_block}

    Personal context (only if helpful and respectful):
    {personal_str}

    Using this information, write 2–3 warm, concrete sentences:
    - Briefly describe what seemed to go well.
    - Point out clearly if calm or focus scores are low (restless, lots of fluctuation).
    - Give 1–2 small, practical suggestions for next time (posture, breath, attention habits, routine).
    Speak directly to them as "you" and end on an encouraging note.
    """.strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\nUser:\n{USER_PROMPT}\n\nCoach:",
        "stream": False,
        "options": {"temperature": 0.5, "num_predict": 160},
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=90)
        if r.status_code == 200:
            return (r.json() or {}).get("response", "").strip()
    except Exception as e:
        return f"(AI Coach unavailable: {e})"

    return "(AI Coach didn’t respond.)"

def init_state():
    defaults = {
        "df": None,
        "file_name": None,
        "summary": None,
        "participant_id": "",
        "task": "",
        "personalization": {},
        "mode": "Visual",
        "ai_coach": False,
        "random_order": ["Visual", "Literal", "LLM"],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_session_for_new_run():
    """Hard reset: clear data + personalization and force fresh widget keys."""
    for k in [
        "df", "file_name", "summary",            # data
        "style", "mode", "ai_coach",             # style 
        "participant_id", "task", "preferred_mode", "ai_coach_enabled",  
        "profile",                                # full personalization dict
    ]:
        st.session_state.pop(k, None)

    # bump a nonce so all input widget keys change -> fields appear empty
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1

    st.session_state["page"] = "Start"

def goto(name: str):
    st.session_state["page"] = name
    st.rerun()

def ensure_ratings_file():
    if not os.path.exists(RATINGS_FILE):
        pd.DataFrame(columns=[
            "timestamp","participant_id","task","file_name",
            "mode","ai_coach","clarity","usefulness","motivation","comments"
        ]).to_csv(RATINGS_FILE, index=False)

def newest_csv(path: str) -> Optional[str]:
    files = glob.glob(os.path.join(path, "*.csv"))
    if not files:
        return None

    def _key(f: str) -> float:
        try:
            if os.name == "nt":
                # Windows: ctime is creation time
                return os.path.getctime(f)
            else:
                # Linux/mac: ctime == metadata change; mtime is usually fine
                return os.path.getmtime(f)
        except OSError:
            return 0.0

    files.sort(key=_key, reverse=True)
    return files[0]

@st.cache_data(show_spinner=False)
def load_csv(file_or_buffer, skiprows: int = 0, cache_buster: float | None = None):
    # Accept both file paths and file-like objects
    if isinstance(file_or_buffer, (str, bytes, io.IOBase)):
        src = file_or_buffer
    else:
        # Streamlit UploadedFile behaves like a buffer
        src = file_or_buffer

    read_kwargs = dict(
    engine="python",
    sep=None,
    comment="%",        # ← ignore OpenBCI comment preface
    skiprows=skiprows,
    )

    # Try reading, falling back through a few common patterns
    try:
        df = pd.read_csv(src, **read_kwargs)
    except Exception:
        # Some BrainFlow files are semicolon-separated on DE locales
        try:
            df = pd.read_csv(src, engine="python", sep=";")
        except Exception:
            # Last resort: comma
            df = pd.read_csv(src, engine="python", sep=",")

    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Normalize column names: strip, remove weird chars, lower minimal
    def _clean_name(s):
        s = str(s).strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace("\uFEFF", "")  # BOM
        return s
    df.columns = [_clean_name(c) for c in df.columns]

    # Replace comma-as-decimal inside object columns BEFORE numeric coercion
    for c in df.columns:
        if df[c].dtype == object:
            # strip whitespace and convert "" to NaN
            s = df[c].astype(str).str.strip()
            s = s.replace({"": np.nan})
            # allow German-style decimal commas
            s = s.str.replace(",", ".", regex=False)
            df[c] = s

    # Try a gentle numeric conversion where possible
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="ignore")
            df[c] = coerced

    return df

def estimate_fs_from_timestamps(df: pd.DataFrame) -> Optional[float]:
    """
    Try to estimate sampling rate (Hz) from a timestamp-like column.

    Handles:
    - Unix seconds (e.g. 1.764e9)
    - ms / µs / ns style timestamps
    Returns a plausible EEG fs in ~20–1000 Hz if possible.
    """
    ts_col = None
    for c in df.columns:
        cname = str(c).lower()
        if "timestamp" in cname or cname in ("time", "t"):
            ts_col = c
            break
    if ts_col is None:
        return None

    s = pd.to_numeric(df[ts_col], errors="coerce").dropna().values
    if s.size < 5:
        return None

    # sort just in case
    s = np.sort(s)
    span = s[-1] - s[0]
    if span <= 0:
        return None

    N = s.size

    def _fs_for(scale: float) -> Optional[float]:
        scaled = s / scale
        dur = scaled[-1] - scaled[0]
        if dur <= 0:
            return None
        return float(N / dur)

    # 1) Assume timestamps are already in seconds (this covers your BrainFlow data)
    fs_sec = _fs_for(1.0)
    if fs_sec is not None and 20.0 <= fs_sec <= 1000.0:
        return fs_sec

    # 2) Try milliseconds
    fs_ms = _fs_for(1e3)
    if fs_ms is not None and 20.0 <= fs_ms <= 1000.0:
        return fs_ms

    # 3) Try microseconds
    fs_us = _fs_for(1e6)
    if fs_us is not None and 20.0 <= fs_us <= 1000.0:
        return fs_us

    # 4) Try nanoseconds
    fs_ns = _fs_for(1e9)
    if fs_ns is not None and 20.0 <= fs_ns <= 1000.0:
        return fs_ns

    # 5) Last-resort fallback (still assuming seconds)
    fs_fallback = N / span
    if 0 < fs_fallback <= 5000:
        return float(fs_fallback)

    return None

def bandpower(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 128 or fs is None or fs <= 0:
        return float("nan")
    w = np.hanning(x.size)
    X = np.fft.rfft((x - np.mean(x)) * w)
    freqs = np.fft.rfftfreq(x.size, d=1.0/fs)
    psd = (np.abs(X) ** 2) / (np.sum(w**2))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return float("nan")
    return float(np.trapezoid(psd[mask], freqs[mask]))

@st.cache_data(show_spinner=False)
def enhanced_summary(df, participant_id: str = "", task_label: str = ""):
    """
    - Detects up to 8 EEG channels by variance, excluding counters, timestamps, and constants.
    - Auto-estimates sampling rate from timestamps when possible.
    - Scales large raw values (OpenBCI counts) to µV and removes DC offset.
    - Uses a combined z-scored multichannel signal for drift (stability).
    - Uses bandpower-derived indices for:
        * calm_score  ~ alpha fraction of (theta+alpha+beta+gamma)
        * focus_score ~ beta / (alpha + theta)
        * movement_score ~ inverse of (beta + gamma) (using all bands)
    - All scores are reported on a 0–100 scale.
    """

    summary = {
        "participant_id": participant_id or "",
        "task_label": task_label or "",
        "calm_score": None,        # 0–100
        "drift_score": None,       # 0–100
        "focus_score": None,       # 0–100
        "movement_score": None,    # 0–100
        "movement_level": None,
        "movement": None,
        "dominant_band": None,
        "duration_sec": None,
    }

    # ---------- 0) Early exit ----------
    if df is None or getattr(df, "empty", False):
        return summary

    df_num = df.copy()

    # ---------- 1) Infer duration from timestamp, if present ----------
    ts_col = None
    for c in df_num.columns:
        cname = str(c).lower()
        if "timestamp" in cname or cname in ("time", "t"):
            ts_col = c
            break

    if ts_col is not None:
        s = _pd.to_numeric(df_num[ts_col], errors="coerce").dropna().values
        if s.size >= 2:
            s = _np.sort(s)
            raw_span = s[-1] - s[0]
            dur = None
            if raw_span > 0:
                # assume seconds first
                dur = float(raw_span)
                # if absurdly long, try ms/ns
                if dur > 60 * 60 * 10:  # >10h
                    if s.max() > 1e10:      # ns
                        dur = float(raw_span / 1e9)
                    elif s.max() > 1e6:     # ms
                        dur = float(raw_span / 1e3)
            summary["duration_sec"] = round(dur, 2) if dur is not None else None

    # ---------- 2) Detect EEG-like channels ----------
    numeric_cols = [
        c for c in df_num.columns
        if _pd.api.types.is_numeric_dtype(df_num[c])
        and not any(k in str(c).lower() for k in ("timestamp", "time", "sample", "marker", "event", "counter"))
    ]
    if not numeric_cols:
        return summary

    df_num = df_num[numeric_cols]

    variances = df_num.var(axis=0, ddof=0)
    var_nonzero = variances[variances > 1e-9]

    EEG_MAX = min(8, len(var_nonzero))
    eeg_cols = list(var_nonzero.sort_values(ascending=False).index[:EEG_MAX])
    if not eeg_cols:
        return summary

    # ---------- 3) Build combined multichannel signal ----------
    sig = df_num[eeg_cols].apply(_pd.to_numeric, errors="coerce")
    sig = sig.replace([_np.inf, -_np.inf], _np.nan).dropna(how="all")
    if sig.shape[0] < 5:
        return summary

    # --- OpenBCI Cyton scaling + DC removal ---
    # If values are extremely large (e.g., |x| > 2000 µV), assume raw ADC counts
    if sig.abs().max().max() > 2000:
        SCALE_UV = 0.02235  # Cyton default (µV per ADC count)
        sig = sig * SCALE_UV

    # Remove per-channel DC offset
    sig = sig - sig.mean(axis=0)

    # Combined z-scored signal
    col_std = sig.std(axis=0, ddof=0).replace(0, _np.nan)
    z = (sig - sig.mean(axis=0)) / (col_std + 1e-9)
    combined = z.mean(axis=1).to_numpy()
    combined = combined[_np.isfinite(combined)]
    if combined.size < 5:
        return summary

    # --- MAD-based drift / instability ---
    def _mad(x):
        x = _np.asarray(x, dtype=float)
        med = _np.median(x)
        return _np.median(_np.abs(x - med))

    diffs = _np.diff(combined)
    mad_signal = _mad(combined) + 1e-9
    mad_diffs = _mad(diffs) + 1e-9

    drift_raw = mad_diffs / (mad_signal + 1e-9)  # higher = more erratic
    drift_raw = float(_np.clip(drift_raw, 0.0, 1.0))
    summary["drift_score"] = drift_raw * 100.0

    # ---------- 4) Sampling rate (fs) ----------
    fs = estimate_fs_from_timestamps(df_num)
    if fs is None or not _np.isfinite(fs) or fs <= 0:
        fs = 250.0  # fallback to typical OpenBCI default

    # ---------- 5) Bandpower & calm/focus/movement ----------
    x = combined
    if x.size >= int(fs * 2):
        # last ~10 seconds or at least 256 samples
        window_samples = int(min(x.size, max(fs * 10, 256)))
        x = x[-window_samples:]
        n = x.size

        w = _np.hanning(n)
        xw = (x - x.mean()) * w

        X = _np.fft.rfft(xw)
        freqs = _np.fft.rfftfreq(n, d=1.0 / fs)
        psd = (abs(X) ** 2) / (w**2).sum()

        def _band(f_lo, f_hi):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if not mask.any():
                return 0.0
            return float(psd[mask].sum())

        # classic EEG bands
        delta = _band(1.0, 4.0)
        theta = _band(4.0, 8.0)
        alpha = _band(8.0, 13.0)
        beta  = _band(13.0, 30.0)
        gamma = _band(30.0, 45.0)

        # --- (A) Relative power including delta (for dominant band + movement) ---
        total_all = delta + theta + alpha + beta + gamma + 1e-9
        rel_all = {
            "delta": delta / total_all,
            "theta": theta / total_all,
            "alpha": alpha / total_all,
            "beta":  beta  / total_all,
            "gamma": gamma / total_all,
        }

        dom = max(rel_all, key=rel_all.get)
        summary["dominant_band"] = dom

        # --- (B) Relative power WITHOUT delta (for calm & focus) ---
        total_td = theta + alpha + beta + gamma + 1e-9
        rel_theta = theta / total_td
        rel_alpha = alpha / total_td
        rel_beta  = beta  / total_td
        rel_gamma = gamma / total_td

        # Calm: alpha fraction (theta+alpha+beta+gamma), alpha≈0.20 → 100
        calm_raw = float(_np.clip(rel_alpha / 0.20, 0.0, 1.0))
        summary["calm_score"] = calm_raw * 100.0

        # Focus: beta / (alpha + theta), ratio≈1 → 100
        ratio = rel_beta / (rel_alpha + rel_theta + 1e-9)
        focus_raw = float(_np.clip(ratio / 1.0, 0.0, 1.0))
        summary["focus_score"] = focus_raw * 100.0

        # Movement: inverse of HF power using delta-inclusive rel_all
        hf_ratio = float(_np.clip(rel_all["beta"] + rel_all["gamma"], 0.0, 1.0))
        movement_score = (1.0 - hf_ratio) * 100.0
        summary["movement_score"] = movement_score

        if movement_score > 75:
            level = "very still"
        elif movement_score > 55:
            level = "mostly still"
        elif movement_score > 35:
            level = "some movement"
        else:
            level = "lots of movement"

        summary["movement_level"] = level
        summary["movement"] = level

    return summary

IGNORE_NAME_PATTERNS = re.compile(r"(time|timestamp|sample|marker|event)", re.I)
LIKELY_SIGNAL_PATTERNS = re.compile(r"(exg|eeg|emg|ecg|gyro|accel|channel|chan|fp\\d|cz|pz|fz)", re.I)

def coerce_numeric_channels(df_in: pd.DataFrame, min_numeric_ratio: float = 0.5):
        """Coerce likely signal columns to numeric. Return (clean_df, numeric_cols)."""
        if df_in is None:
            return df_in, []
        clean = df_in.copy()
        numeric_cols = []
        for col in clean.columns:
            name = str(col)
            skip_by_name = bool(IGNORE_NAME_PATTERNS.search(name))
            s = clean[col]

            if pd.api.types.is_numeric_dtype(s):
                if s.notna().sum() > 0 and not skip_by_name:
                    numeric_cols.append(col)
                continue

            if s.dtype == object:
                s_num = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
                ratio = s_num.notna().mean()
                if ratio >= min_numeric_ratio or LIKELY_SIGNAL_PATTERNS.search(name):
                    clean[col] = s_num
                    if not skip_by_name:
                        numeric_cols.append(col)

        if not numeric_cols:
            for col in clean.columns:
                if pd.api.types.is_numeric_dtype(clean[col]) and clean[col].notna().sum() > 0:
                    numeric_cols.append(col)
        return clean, numeric_cols    

def get_session_scores(df: pd.DataFrame):
    """
    Returns (calm_pct, focus_pct, stability_0to1, df_clean, to_plot).
    - calm_pct / focus_pct prefer enhanced_summary values.
    - If calm_score is missing, fall back to stability-based calm from multichannel signal.
    - If focus_score is missing, returns 50.0 by default.
    """
    summary = st.session_state.get("summary") or {}
    calm_pct = summary.get("calm_score")
    focus_pct = summary.get("focus_score")

    df_clean, numeric_cols = coerce_numeric_channels(df)

    stability = 0.5
    to_plot: list[str] = []

    if numeric_cols:
        to_plot = numeric_cols[:4]
        sig = df_clean[to_plot].apply(pd.to_numeric, errors="coerce")
        sig = sig.replace([np.inf, -np.inf], np.nan).dropna(how="all")

        if sig.shape[0] >= 5:
            col_std = sig.std(axis=0, ddof=0).replace(0, np.nan)
            z = (sig - sig.mean(axis=0)) / (col_std + 1e-9)
            combined = z.mean(axis=1).to_numpy()
            stability = float(np.clip(compute_stability_score(combined), 0.0, 1.0))

    if calm_pct is None:
        calm_pct = stability * 100.0
    if focus_pct is None:
        focus_pct = 50.0

    calm_pct = float(np.clip(calm_pct, 0.0, 100.0))
    focus_pct = float(np.clip(focus_pct, 0.0, 100.0))

    return calm_pct, focus_pct, stability, df_clean, to_plot

def ratings_append(row: Dict[str, Any]):
    ensure_ratings_file()
    df = pd.read_csv(RATINGS_FILE)
    df.loc[len(df)] = row
    df.to_csv(RATINGS_FILE, index=False)

def render_timeseries_light(df_in: pd.DataFrame, cols: list):
    """Light, soft-styled multi-line chart."""
    df_plot = df_in[cols].reset_index().rename(columns={"index":"t"})
    long = df_plot.melt(id_vars="t", var_name="channel", value_name="value")

    base = alt.Chart(long).mark_line(strokeWidth=1.5).encode(
        x=alt.X("t:Q", axis=alt.Axis(title=None, grid=True, gridOpacity=0.15,
                                     labelOpacity=0.7, tickOpacity=0.6,
                                     labelColor="#174C4F", tickColor="#CFE3E0", domainColor="#CFE3E0")),
        y=alt.Y("value:Q", axis=alt.Axis(title=None, grid=True, gridOpacity=0.15,
                                        labelOpacity=0.7, tickOpacity=0.6,
                                        labelColor="#174C4F", tickColor="#CFE3E0", domainColor="#CFE3E0")),
        color=alt.Color("channel:N",
                        scale=alt.Scale(scheme="tealblues"),
                        legend=alt.Legend(title=None, labelColor="#174C4F"))
    ).properties(height=260, background="#F7FCFA")
    return base

init_state()

page = st.session_state["page"]

# -----------------------------
# Start Page
# -----------------------------
if page == "Start":
    st.markdown(
        """
        <style>
        .stApp { background: #F7FCFA; }

        .welcome-title{
          font-family: 'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
          font-weight: 600; font-size: 2.3rem; color:#174C4F; margin-bottom:.6rem; text-align:center;
        }
        .subtitle{ font-family:'Poppins',system-ui; font-size:1.05rem; color:#174C4F; opacity:.9; margin-bottom:.3rem; text-align:center;}
        .caption{ font-size:.95rem; color:#174C4F; opacity:.75; margin-bottom:1.2rem; text-align:center;}

        /* Center Streamlit button & status text */
        .center-container { display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; }

        div.stButton > button{
        padding:.8rem 2.2rem !important;      
        width: 240px !important;               
        border-radius:16px !important;
        font-weight:600 !important;
        font-size:1.05rem !important;
        color:#fff !important;
        background:#4AA3A2 !important;
        border:none !important;
        box-shadow:0 6px 14px rgba(0,0,0,.15);
        transition:all .25s ease-in-out;
        }

        div.stButton > button:hover{
        transform:translateY(-2px) scale(1.03);
        background:#3E8E8D !important;
        box-shadow:0 8px 16px rgba(0,0,0,.22);
        }

        /* Greyed-out disabled button */
        div.stButton > button:disabled {
          background: #C9D3D3 !important;
          color: #6B7A7A !important;
          box-shadow: none !important;
          cursor: not-allowed !important;
          transform: none !important;
        }

        /* Fade-in + loader */
        .fadein{ animation:fadein .35s ease-in both; }
        @keyframes fadein{ from{opacity:0; transform:translateY(-4px);} to{opacity:1; transform:translateY(0);} }
        .loader{ display:inline-block; width:18px; height:18px; border:3px solid #E5E7EB;
                 border-top:3px solid #4AA3A2; border-radius:50%; animation:spin .9s linear infinite;
                 vertical-align:middle; margin-right:8px; }
        @keyframes spin{100%{transform:rotate(360deg);} }

        /* Expander style - always teal, centered */
        div[data-testid="stExpander"]{
          background:#D6EFEF !important; border:1px solid #A7D7D6 !important; border-radius:12px !important;
          width:70%; margin:3rem auto; color:#174C4F !important;
        }
        div[data-testid="stExpander"] div[role="button"] p,
        div[data-testid="stExpander"] div[role="button"] span{ color:#174C4F !important; font-weight:600; }
        div[data-testid="stExpander"] *{ color:#174C4F !important; }
        div[data-testid="stExpander"] div[data-testid="stFileUploaderDropzone"]{
          background:#EAF6F6 !important; border:1px dashed #A7D7D6 !important;
        }
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True
    )

    # ---------- Layout ----------
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown("<div class='welcome-title'>Welcome</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Your session’s complete — let’s explore how your mind performed.</div>", unsafe_allow_html=True)
        st.markdown("<div class='caption'>When you’re ready, press below to view your insights and personalized feedback.</div>", unsafe_allow_html=True)

        # scanning indicator
        status_ph = st.empty()
        status_ph.markdown("<div class='caption'><span class='loader'></span>Scanning for your latest session…</div>", unsafe_allow_html=True)

        # find latest file
        path = WATCH_DEFAULT
        latest = newest_csv(path)

        if latest:
            file_name = os.path.basename(latest)
            disabled = False
            msg_html = "<div class='fadein' style='font-size:0.95rem; color:#2E8B57; margin-top:.8rem;'>✅ <b>Found it!</b></div>"
        else:
            file_name = None
            disabled = True
            msg_html = "<div class='fadein' style='font-size:0.95rem; color:#D97706; margin-top:.8rem;'>⚠️ <b>Oh no!</b> There's nothing here yet — try saving your session or use the manual upload below.</div>"

        status_ph.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # button + message
        btn_l, btn_c, btn_r = st.columns([1, 1, 1])
        with btn_c:
            clicked = st.button("✨ Get Results", disabled=disabled)
            st.markdown(msg_html.replace("margin-top:.8rem;", "margin-top:.8rem; text-align:center;"), unsafe_allow_html=True)

        if clicked and latest:
            try:
                mtime = os.path.getmtime(latest)
            except Exception:
                mtime = time.time()

            df = load_csv(latest, cache_buster=mtime)
            st.session_state["df"] = df
            st.session_state["file_name"] = file_name
            st.session_state["summary"] = enhanced_summary(
                df,
                st.session_state.get("participant_id", ""),
                st.session_state.get("task", ""),
            )
            st.session_state["page"] = "Personalization"
            st.rerun()

    # ---------- Expander (manual upload fallback) ----------
    with st.expander("Upload a file manually"):
        up = st.file_uploader("Upload OpenBCI CSV", type=["csv"])
        if up is not None:
            if st.button("Analyze uploaded file"):
                # each upload is treated as new
                df = load_csv(up, cache_buster=time.time())
                st.session_state["df"] = df
                st.session_state["file_name"] = getattr(up, "name", "uploaded.csv")
                st.session_state["summary"] = enhanced_summary(
                    df,
                    st.session_state.get("participant_id", ""),
                    st.session_state.get("task", ""),
                )
                st.session_state["page"] = "Personalization"
                st.rerun()

# -----------------------------
# Personalization Page
# -----------------------------
if page == "Personalization":
    st.markdown("""
        <style>
        .stApp { background:#F7FCFA; }
        .pers-title{
            font-family:'Poppins',system-ui; font-weight:600;
            font-size:1.8rem; color:#174C4F; text-align:left;
        }
        .pers-sub{
            font-family:'Poppins',system-ui; color:#174C4F; opacity:.85;
            margin:.25rem 0 1rem 0; text-align:left;
        }
        .stTextInput label,
        .stSelectbox label,
        .stTextArea label,
        div[data-testid="stMultiSelect"] label {
            color:#174C4F !important; font-weight:600;
        }
        div[data-testid="stCheckbox"] *,
        div[data-testid="stCheckbox"] label {
            color:#174C4F !important; font-weight:600;
        }
        div.stButton > button[kind="primary"]{
            background:#4AA3A2 !important; color:#fff !important; border:none !important;
            border-radius:14px !important; padding:.6rem 1.3rem !important; font-weight:600 !important;
            box-shadow:0 6px 14px rgba(0,0,0,.12);
        }
        div.stButton > button[kind="primary"]:hover{
            background:#3E8E8D !important; transform:translateY(-2px);
        }
        div.stButton > button:disabled{
            background:#C9D3D3 !important; color:#6B7A7A !important; box-shadow:none !important;
            cursor:not-allowed !important; transform:none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>

        /* ---------------------------------------------------
        FORCE ALL INPUT FIELDS TO USE LIGHT COLORS
        regardless of OS/system dark mode
        --------------------------------------------------- */

        /* Selectbox + dropdown */
        div[data-baseweb="select"] > div {
            background: #FFFFFF !important;
            border: 1px solid #CFE3E0 !important;
            color: #174C4F !important;
            border-radius: 10px !important;
        }
        div[data-baseweb="popover"] {
            background: #FFFFFF !important;
            color: #174C4F !important;
            border-radius: 10px !important;
        }

        /* Selectbox items */
        ul[role="listbox"] li {
            background: #FFFFFF !important;
            color: #174C4F !important;
        }
        ul[role="listbox"] li:hover {
            background: #EAF4F2 !important;
        }

        /* Text input + text area */
        input[type="text"], textarea {
            background: #FFFFFF !important;
            color: #174C4F !important;
            border: 1px solid #CFE3E0 !important;
            border-radius: 10px !important;
        }

        /* Checkbox text */
        div[data-testid="stCheckbox"] > label > div {
            color: #174C4F !important;
        }

        /* Make placeholder text readable */
        input::placeholder, textarea::placeholder {
            color: #8AA5A3 !important;
            opacity: 1 !important;
        }

        </style>
        """, unsafe_allow_html=True)

    # --- ensure profile exists ---
    if "profile" not in st.session_state:
        st.session_state["profile"] = {
            "task": "",
            "age_range": "",
            "gender": "",
            "occupation": "",
            "eeg_experience": "",
            "primary_goal": "",
            "consent": False,
        }
    prof = st.session_state["profile"]

    nonce = st.session_state.get("nonce", 0)

    hdr_l, hdr_c, hdr_r = st.columns([1, 8, 2])

    with hdr_l:
        st.markdown("""
            <style>
            div[data-testid="back_button"] > button {
                background: transparent !important;
                color: #174C4F !important;
                border: none !important;
                font-size: 1.4rem !important;
                font-weight: 700 !important;
                padding: 0 !important;
                margin-top: 0.3rem !important;
            }
            div[data-testid="back_button"] > button:hover {
                color: #3E8E8D !important;
                transform: translateX(-2px);
            }
            </style>
        """, unsafe_allow_html=True)
        if st.button("←", key=f"back_to_start_{nonce}", help="Go back to start"):
            st.session_state["page"] = "Start"
            st.rerun()

    with hdr_c:
        st.markdown("<div class='pers-title'>Personalize your Feedback</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='pers-sub'>These fields help personalize your experience.</div>",
            unsafe_allow_html=True
        )

    # --- dropdown options (no typing except occupation) ---
    age_opts    = ["", "Under 18", "18–29", "30–39", "40–49", "50–59", "60+"]
    gender_opts = ["", "Female", "Male", "Non-binary", "Prefer not to say", "Other"]
    eeg_opts    = ["", "None", "Beginner", "Intermediate", "Advanced"]
    goal_opts   = ["", "Improve focus", "Reduce stress", "Sustain attention", "Better sleep", "Other"]

    task_opts = [
        "",
        "Reading",
        "Logic puzzle",
        "Math problem",
        "Drawing / Doodling",
        "Playing a game",
        "Listening to music",
        "Meditating"
    ]

    def _safe_index(options, value):
        return options.index(value) if value in options else 0

    # --- task — MANDATORY ---
    st.markdown(
        "<div style='font-weight:600; color:#174C4F; margin-bottom:4px;'>"
        "What were you doing in your session? "
        "<span style='color:#D9534F;'>*</span>"
        "</div>",
        unsafe_allow_html=True
    )

    prof["task"] = st.selectbox(
        "Session activity (required)",
        task_opts,
        index=_safe_index(task_opts, prof.get("task", "")),
        key=f"task_{nonce}",
        label_visibility="collapsed",
    )

    st.session_state["task"] = prof.get("task", "")

    # Red warning text if empty
    if not prof.get("task"):
        st.markdown(
            "<div style='color:#D9534F; font-size:0.85rem; margin-top:-4px;'>"
            "This field is required.</div>",
            unsafe_allow_html=True,
        )

    prof["age_range"] = st.selectbox(
        "Age Range",
        age_opts,
        index=_safe_index(age_opts, prof.get("age_range", "")),
        key=f"age_range_{nonce}",
    )
    prof["gender"] = st.selectbox(
        "Gender",
        gender_opts,
        index=_safe_index(gender_opts, prof.get("gender", "")),
        key=f"gender_{nonce}",
    )
    prof["occupation"] = st.text_input(
        "Occupation",
        value=prof.get("occupation", ""),
        key=f"occupation_{nonce}",
    )
    prof["eeg_experience"] = st.selectbox(
        "Prior EEG Experience",
        eeg_opts,
        index=_safe_index(eeg_opts, prof.get("eeg_experience", "")),
        key=f"eeg_experience_{nonce}",
    )
    prof["primary_goal"] = st.selectbox(
        "Primary Goal",
        goal_opts,
        index=_safe_index(goal_opts, prof.get("primary_goal", "")),
        key=f"primary_goal_{nonce}",
    )

    prof["consent"] = st.checkbox(
        "I consent to the anonymous use of my responses for research analysis.",
        value=bool(prof.get("consent", False)),
        key=f"consent_checkbox_{nonce}",
    )

    # --- Next button ---
    c1, c2, c3 = st.columns([3, 2, 3])
    with c2:
        can_continue = bool(prof["consent"] and prof.get("task"))
        if st.button(
            "Next →",
            key=f"next_from_personalization_btn_{nonce}",
            disabled=not can_continue,
            type="primary",
        ):
            st.session_state["personalization_skipped"] = False

            # task is given
            df_cur = st.session_state.get("df")
            if df_cur is not None and not getattr(df_cur, "empty", True):
                st.session_state["summary"] = enhanced_summary(
                    df_cur,
                    st.session_state.get("participant_id", ""),
                    st.session_state.get("task", prof.get("task", "")),   
                )

            st.session_state["page"] = "Choose Style"
            st.rerun()

# -----------------------------
# Choose Style Page
# -----------------------------
if page == "Choose Style":
    st.markdown("""
        <style>
        .stApp { background:#F7FCFA; }

        /* Title and subtitle */
        .cs-title{
            font-family:'Poppins',system-ui;font-weight:600;
            font-size:1.9rem;color:#174C4F;text-align:left;
        }
        .cs-sub{
            font-family:'Poppins',system-ui;color:#174C4F;
            opacity:.85;margin:.35rem 0 1.2rem;text-align:left;
        }

        /* Back arrow */
        div[data-testid="back_button_style"] > button {
            background: transparent !important;
            color: #174C4F !important;
            border: none !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            padding: 0 !important;
            margin-top: 0.3rem !important;
        }
        div[data-testid="back_button_style"] > button:hover {
            color: #3E8E8D !important;
            transform: translateX(-2px);
        }

        /* Radio + Checkbox styling — bigger, better spacing */
        div[data-testid="stRadio"] label p,
        div[data-testid="stCheckbox"] label p {
            font-size: 1.05rem !important;
            color: #174C4F !important;
        }
        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label {
            padding: 0.4rem 0 !important;
        }
        div[data-testid="stRadio"] input,
        div[data-testid="stCheckbox"] input {
            transform: scale(1.3) !important;
            margin-right: 10px !important;
        }
        div[data-testid="stRadio"] > div {
            row-gap: 0.6rem !important;
        }

        /* Ensure everything visible */
        div[data-testid="stRadio"] *,
        div[data-testid="stCheckbox"] * {
            color:#174C4F !important;
            opacity:1 !important;
        }

        /* Primary buttons */
        div.stButton > button[kind="primary"]{
            background:#4AA3A2 !important; color:#fff !important; border:none !important;
            border-radius:14px !important; padding:.7rem 1.4rem !important; font-weight:600 !important;
            font-size:1.05rem !important;
            box-shadow:0 6px 14px rgba(0,0,0,.12);
        }
        div.stButton > button[kind="primary"]:hover{
            background:#3E8E8D !important; transform:translateY(-2px);
        }
        div.stButton > button:disabled{
            background:#C9D3D3 !important; color:#6B7A7A !important;
            box-shadow:none !important; cursor:not-allowed !important; transform:none !important;
        }

        /* AI checkmark style */
        .ai-check {
            color:#174C4F;
            font-family:'Poppins',system-ui;
            font-size:1.05rem;
            display:flex;
            align-items:center;
            gap:0.5rem;
            margin-top:0.4rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- state ---
    if "style" not in st.session_state:
        st.session_state["style"] = {"mode": "", "ai_coach": False}
    style = st.session_state["style"]

    # --- header with back + title ---
    hdr_l, hdr_c, _ = st.columns([1, 8, 1])
    with hdr_l:
        st.markdown('<div data-testid="back_button_style"></div>', unsafe_allow_html=True)
        if st.button("←", key="back_to_personalization_style", help="Go back to Personalization"):
            st.session_state["page"] = "Personalization"
            st.rerun()
    with hdr_c:
        st.markdown("<div class='cs-title'>Choose your feedback style</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='cs-sub'>Pick how you'd like to experience your session results.</div>",
            unsafe_allow_html=True
        )

    # --- style choice ---
    options = ["Visual", "Literal", "LLM"]
    labels = {
        "Visual": "Visual Progress Report",
        "Literal": "Literal Reflection",
        "LLM": "AI Coach — Personalized Guidance by AI"
    }

    idx = options.index(style["mode"]) if style["mode"] in options else None
    chosen = st.radio(
        "Output Style:",
        options,
        index=idx,
        format_func=lambda x: labels[x],
        horizontal=False,
        key="style_mode_radio"
    )

    # --- detect if mode changed ---
    prev_mode = style.get("mode", "")
    st.session_state["style"]["mode"] = chosen

    if prev_mode == "LLM" and chosen != "LLM":
        # Reset AI toggle when leaving LLM
        st.session_state["style"]["ai_coach"] = False

    st.write("")

    # --- AI coach control ---
    mode = st.session_state["style"]["mode"]

    if mode == "LLM":
        # Automatically enabled + show checkmark
        st.session_state["style"]["ai_coach"] = True
        st.markdown(
            "<div class='ai-check'>✅ AI-Coach is automatically included with this style.</div>",
            unsafe_allow_html=True
        )
    else:
        ai_val = st.checkbox(
            "Enable AI-Coach",
            value=st.session_state["style"]["ai_coach"],
            key="ai_coach_checkbox_style",
            help="Short coaching tips alongside your chosen feedback style."
        )
        st.session_state["style"]["ai_coach"] = ai_val

    st.write("")

    # --- Next button ---
    left_spacer, next_col = st.columns([8, 2])
    with next_col:
        can_proceed = bool(st.session_state["style"]["mode"])
        if st.button("Next →", key="next_from_style", type="primary", disabled=not can_proceed):
            st.session_state["page"] = "Display"
            st.rerun()

# -----------------------------
# Display Page
# -----------------------------
if page == "Display":
    # ---- page styles ----
    st.markdown("""
        <style>
        .stApp { background:#F7FCFA; }
        .disp-title{
            font-family:'Poppins',system-ui;font-weight:600;
            font-size:1.9rem;color:#174C4F;text-align:left;
        }
        .disp-sub{
            font-family:'Poppins',system-ui;color:#174C4F;
            opacity:.85;margin:.35rem 0 1.2rem;text-align:left;
        }
                
        div[data-testid="stSpinner"] p,
        div[data-testid="stSpinner"] {
            color: #174C4F !important;
            opacity: 1 !important;
            font-family: 'Poppins', system-ui !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
        }

        /* Make all markdown on this page readable */
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
        .stMarkdown strong, .stMarkdown em {
          color:#174C4F !important;
          opacity:1 !important;
        }

        /* Back arrow look */
        div[data-testid="back_button_display"] > button {
            background: transparent !important;
            color: #174C4F !important;
            border: none !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            padding: 0 !important;
            margin-top: 0.3rem !important;
        }
        div[data-testid="back_button_display"] > button:hover {
            color: #3E8E8D !important;
            transform: translateX(-2px);
        }

        /* Primary buttons */
        div.stButton > button[kind="primary"]{
            background:#4AA3A2 !important; color:#fff !important; border:none !important;
            border-radius:14px !important; padding:.7rem 1.4rem !important; font-weight:600 !important;
            font-size:1.05rem !important; box-shadow:0 6px 14px rgba(0,0,0,.12);
        }
        div.stButton > button[kind="primary"]:hover{
            background:#3E8E8D !important; transform:translateY(-2px);
        }
        div.stButton > button:disabled{
            background:#C9D3D3 !important; color:#6B7A7A !important;
            box-shadow:none !important; cursor:not-allowed !important; transform:none !important;
        }

        /* Star animation */
        @keyframes popIn {
            0%   { transform: scale(0.6); opacity: 0; }
            60%  { transform: scale(1.12); opacity: 1; }
            85%  { transform: scale(0.96); }
            100% { transform: scale(1.00); }
        }
        @keyframes gentleBob {
            0%   { transform: translateY(0px); }
            50%  { transform: translateY(-2px); }
            100% { transform: translateY(0px); }
        }
        .stars-hero { animation: popIn 550ms ease-out both; }
        .stars-hero .shine {
            font-size: 4.6rem;
            color: #FFD700;
            text-shadow: 0 0 15px rgba(255,215,0,0.6),
                         0 0 30px rgba(255,200,0,0.4);
            letter-spacing: 0.3rem;
            line-height: 1;
            display: inline-block;
            animation: gentleBob 2.5s ease-in-out 700ms infinite;
        }

        /* Literal mode: torn notebook page */
        .lit-card {
            position: relative;
            background: #FFFCF4;
            padding: 1.6rem 1.8rem 1.4rem;
            margin-top: 1.4rem;
            color: #174C4F;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;

            border-radius: 12px;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;

            box-shadow: 0 14px 28px rgba(0,0,0,.11);
            border-left: 1px solid #E2D6BD;
            border-right: 1px solid #E2D6BD;

            transform: rotate(-0.35deg);
        }

        /* torn top edge */
        .lit-card::before {
            content: "";
            position: absolute;
            left: -10px;
            right: -10px;
            top: -20px;
            height: 32px;
            background:
              radial-gradient(circle at 12px 26px, #FFFCF4 0 12px, transparent 12px),
              radial-gradient(circle at 52px 18px, #FFFCF4 0 14px, transparent 14px),
              radial-gradient(circle at 94px 24px, #FFFCF4 0 13px, transparent 13px),
              radial-gradient(circle at 140px 16px, #FFFCF4 0 14px, transparent 14px),
              radial-gradient(circle at 186px 23px, #FFFCF4 0 13px, transparent 13px);
            background-size: 190px 30px;
            background-repeat: repeat-x;
            border-bottom: 1px solid #E2D6BD;
        }

        /* torn bottom edge */
        .lit-card::after {
            content: "";
            position: absolute;
            left: -10px;
            right: -10px;
            bottom: -22px;
            height: 34px;
            background:
              radial-gradient(circle at 18px 4px, #FFFCF4 0 12px, transparent 12px),
              radial-gradient(circle at 64px 10px, #FFFCF4 0 14px, transparent 14px),
              radial-gradient(circle at 110px 6px, #FFFCF4 0 13px, transparent 13px),
              radial-gradient(circle at 158px 11px, #FFFCF4 0 14px, transparent 14px),
              radial-gradient(circle at 204px 5px, #FFFCF4 0 13px, transparent 13px);
            background-size: 200px 32px;
            background-repeat: repeat-x;
            border-top: 1px solid #E2D6BD;
            box-shadow: 0 10px 16px rgba(0,0,0,.18);
        }

        .lit-header {
            font-family: 'Poppins', system-ui;
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: .4rem;
            letter-spacing: .16em;
            text-transform: uppercase;
            color: #9A8355;
        }

        .lit-body {
            font-size: 1.05rem;
            line-height: 1.7;
            white-space: pre-line;   /* keep poem line breaks */
            text-align: justify;
        }

        .lit-quote::before {
            content: "“";
            font-size: 2.6rem;
            line-height: 0;
            position: relative;
            top: 1rem;
            margin-right: .25rem;
            color: #D0AF6D;
            opacity: .8;
        }

        .lit-foot {
            font-size: .9rem;
            opacity: .75;
            margin-top: .9rem;
            text-align: right;
            font-style: italic;
            color: #8C744A;
        }
        .tech-card {
            margin-top: 2.2rem;
            max-width: 900px;
            margin-left:auto;
            margin-right:auto;
            background:#EAF4F2;
            border-radius:14px;
            border:1px solid #CFE3E0;
            padding:1.1rem 1.3rem 1.0rem;
            box-shadow:0 10px 22px rgba(0,0,0,.07);
            color:#174C4F;
        }
        .tech-title {
            font-family:'Poppins',system-ui;
            font-weight:600;
            font-size:.98rem;
            text-transform:uppercase;
            letter-spacing:.16em;
            margin-bottom:.3rem;
            color:#4A7F7C;
        }
        .tech-body {
            font-size:.97rem;
            line-height:1.6;
        }
        .tech-body ul {
            margin:.2rem 0 0;
            padding-left:1.2rem;
        }
        .tech-body li {
            margin-bottom:.15rem;
        }
                
        <style>
        /* Robot avatar*/
        .robot-talk-wrapper {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin: 1.2rem auto;
        max-width: 900px;
        }

        /* Robot icon circle */
        .robot-icon {
        width: 48px;
        height: 48px;
        background: #E8F7F3;
        border: 2px solid #CFE3E0;
        border-radius: 50%;
        display:flex;
        justify-content:center;
        align-items:center;
        font-size: 1.6rem;
        box-shadow: 0 4px 10px rgba(0,0,0,.07);
        }

        /* Speech bubble */
        .llm-bubble {
        position: relative;
        background: #FFFFFF;
        padding: 1.0rem 1.3rem;
        border-radius: 18px;
        font-family:'Poppins',system-ui;
        font-size: 1.05rem;
        color:#174C4F;
        line-height:1.55;
        border:1px solid #D4E7E4;
        box-shadow:0 8px 20px rgba(0,0,0,.06);
        flex-grow:1;
        }

        /* Speech bubble arrow tail */
        .llm-bubble::before {
        content:"";
        position:absolute;
        left:-14px;
        top:20px;
        width:0;
        height:0;
        border-top:10px solid transparent;
        border-bottom:10px solid transparent;
        border-right:14px solid #FFFFFF;
        filter: drop-shadow(-2px 2px 2px rgba(0,0,0,.06));
        }

        /* Ensure all inner text stays colored correctly */
        .llm-bubble * {
        color:#174C4F !important;
        opacity:1 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ---- header: back + title ----
    left, mid = st.columns([1, 8])
    with left:
        st.markdown('<div data-testid="back_button_display"></div>', unsafe_allow_html=True)
        if st.button("←", key="back_to_style_display", help="Go back to Choose Style"):
            st.session_state["page"] = "Choose Style"
            st.rerun()
    with mid:
        st.markdown("<div class='disp-title'>Your Session Results</div>", unsafe_allow_html=True)
        st.markdown("<div class='disp-sub'>Here’s what your brain activity revealed.</div>", unsafe_allow_html=True)

    # ---------- Self-healing fetch ----------
    df = st.session_state.get("df", None)
    if df is None or getattr(df, "empty", True):
        latest = newest_csv(WATCH_DEFAULT)
        if latest:
            try:
                try:
                    mtime = os.path.getmtime(latest)
                except Exception:
                    mtime = time.time()

                df = load_csv(latest, cache_buster=mtime)
                st.session_state["df"] = df
                st.session_state["file_name"] = os.path.basename(latest)
            except Exception as e:
                st.warning(f"Couldn’t load session file: {e}")
                df = None

    if df is not None and not df.empty:
        st.session_state["summary"] = enhanced_summary(
            df,
            st.session_state.get("participant_id", ""),
            st.session_state.get("task", ""),
        )

    # ---- current style choice ----
    style = st.session_state.get("style", {"mode": "Visual", "ai_coach": False})
    mode = style.get("mode", "Visual")
    ai_on = style.get("ai_coach", False)

    if df is None or getattr(df, "empty", True):
        st.warning("⚠️ No data available. Please go to **Start** and press **Get Results** once.")
    else:
        if mode == "Visual":
            calm_pct, focus_pct, stability, df_clean, to_plot = get_session_scores(df)

            if not to_plot:
                st.info("No numeric channels detected to plot.")
            else:
                calm_score = calm_pct / 100.0
                focus_score = focus_pct / 100.0

                color_hex = score_to_color(calm_score)
                badge = calm_badge(calm_score)
                focus_label = focus_band(focus_score)

                # --- Splatter “mind orb” CSS (big, glowing, animated) ---
                st.markdown(
                    f"""
                    <style>
                    ...  # ← keep your existing CSS here, unchanged
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # --- Splatter “mind orb” CSS (big, glowing, animated) ---
                st.markdown(
                    f"""
                    <style>
                    .mind-orb-wrapper {{
                        display:flex;
                        justify-content:center;
                        align-items:center;
                        margin-top:1.6rem;
                        margin-bottom:.9rem;
                    }}

                    /* base blob */
                    .calm-splatter {{
                        width: 150px;
                        height: 150px;
                        display:inline-block;
                        vertical-align:middle;
                        background: radial-gradient(
                            circle at 35% 30%,
                            {color_hex}DD,
                            #A0E7DB88,
                            #F3FFFB55
                        );
                        filter: drop-shadow(0 0 20px {color_hex}88)
                                drop-shadow(0 0 40px {color_hex}55);
                        transform-origin: center;
                    }}

                    /* Very irregular, splashy outline for stormy sessions */
                    .splatter-stormy {{
                        clip-path: polygon(
                            50% 0%,
                            63% 6%,
                            76% 3%,
                            88% 10%,
                            98% 24%,
                            100% 40%,
                            93% 56%,
                            100% 70%,
                            90% 83%,
                            78% 92%,
                            61% 96%,
                            50% 100%,
                            37% 95%,
                            24% 98%,
                            13% 90%,
                            4% 75%,
                            0% 60%,
                            3% 42%,
                            0% 28%,
                            6% 14%,
                            18% 5%,
                            32% 3%
                        );
                        animation: splatter-pulse-big 2.6s ease-in-out infinite;
                    }}

                    /* Medium blob for settling / steadying */
                    .splatter-settling,
                    .splatter-steadying {{
                        clip-path: polygon(
                            50% 0%,
                            66% 6%,
                            80% 18%,
                            92% 34%,
                            96% 50%,
                            91% 66%,
                            80% 82%,
                            64% 94%,
                            50% 100%,
                            34% 95%,
                            20% 84%,
                            9% 68%,
                            4% 52%,
                            7% 34%,
                            16% 20%,
                            30% 8%
                        );
                        animation: splatter-pulse-medium 3.1s ease-in-out infinite;
                    }}

                    /* Almost round for smooth / flow */
                    .splatter-smooth,
                    .splatter-flow {{
                        clip-path: circle(46% at 50% 50%);
                        animation: splatter-pulse-soft 3.6s ease-in-out infinite;
                    }}

                    @keyframes splatter-pulse-big {{
                        0%   {{ transform: scale(0.88) rotate(-2deg); }}
                        50%  {{ transform: scale(1.06) rotate(2deg);  }}
                        100% {{ transform: scale(0.88) rotate(-2deg); }}
                    }}
                    @keyframes splatter-pulse-medium {{
                        0%   {{ transform: scale(0.92); }}
                        50%  {{ transform: scale(1.04); }}
                        100% {{ transform: scale(0.92); }}
                    }}
                    @keyframes splatter-pulse-soft {{
                        0%   {{ transform: scale(0.96); }}
                        50%  {{ transform: scale(1.02); }}
                        100% {{ transform: scale(0.96); }}
                    }}

                    .calm-meta {{
                        text-align:left;
                        color:#174C4F;
                        margin-top:1.2rem;
                    }}
                    .calm-meta .badge {{
                        display:inline-block;
                        padding:.3rem .8rem;
                        border-radius:999px;
                        font-weight:600;
                        font-size:.9rem;
                        color:#174C4F;
                        background:#EAF4F2;
                        border:1px solid #CFE3E0;
                        margin-bottom:.25rem;
                    }}
                    .calm-meta .score-line {{
                        font-size:1.1rem;
                        font-weight:500;
                    }}
                    .calm-meta .hint {{
                        font-size:.9rem;
                        opacity:.8;
                        margin-top:.15rem;
                    }}

                    .focus-card {{
                        margin-top:.7rem;
                        font-size:.95rem;
                    }}
                    .focus-title {{
                        font-weight:600;
                        margin-bottom:.15rem;
                    }}
                    .focus-bar-outer {{
                        width:100%;
                        height:10px;
                        border-radius:999px;
                        background:#EAF4F2;
                        overflow:hidden;
                        margin-bottom:.25rem;
                    }}
                    .focus-bar-inner {{
                        height:100%;
                        border-radius:999px;
                        background: linear-gradient(90deg, #F0AD4E, #4AA3A2);
                    }}
                    .focus-hint {{
                        font-size:.88rem;
                        opacity:.85;
                    }}

                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # --- Orb + calm + focus text ---
                orb_col, txt_col = st.columns([3, 4])

                with orb_col:
                    st.markdown(
                        f"""
                        <div class="mind-orb-wrapper">
                            <div class="calm-splatter splatter-{badge}"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with txt_col:
                    st.markdown(
                        f"""
                        <div class="calm-meta">
                            <div class="badge">state: {badge}</div>
                            <div class="score-line">calm alignment: <b>{calm_pct:.0f}%</b></div>
                            <div class="hint">
                                Higher % means smoother, more stable brainwaves during this session.<br>
                                Jagged, splashy edges = more restless; round and soft = more settled.
                            </div>
                            <div class="focus-card">
                                <div class="focus-title">
                                    Focus signal: <b>{focus_label}</b> ({focus_pct:.0f}%)
                                </div>
                                <div class="focus-bar-outer">
                                    <div class="focus-bar-inner" style="width:{focus_pct:.0f}%;"></div>
                                </div>
                                <div class="focus-hint">
                                    Higher % suggests more sustained, task-aligned attention
                                    (less mental wandering).
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # --- signal chart ---
                st.markdown(
                    "<div style='margin-top:1.4rem; color:#174C4F; opacity:.85;'>Signal over time</div>",
                    unsafe_allow_html=True,
                )

                # Normalize channels so they’re comparable and the legend stays readable
                vis = df_clean[to_plot].apply(pd.to_numeric, errors="coerce")
                vis = vis.replace([np.inf, -np.inf], np.nan).dropna(how="all")

                if vis.empty:
                    st.caption("Not enough clean signal samples to show a meaningful chart.")
                else:

                    # Turn first 4 channels into a tidy table
                    plot_df = df_clean[to_plot].reset_index().rename(columns={"index": "t"})

                    # Map original column names → "Sensor 1..4"
                    chan_labels = {col: f"Sensor {i+1}" for i, col in enumerate(to_plot)}
                    plot_df = plot_df.rename(columns=chan_labels)

                    long_df = plot_df.melt(id_vars="t", var_name="channel", value_name="value")

                    line_chart = (
                        alt.Chart(long_df)
                        .mark_line(strokeWidth=1.5)
                        .encode(
                            x=alt.X("t:Q", title="Time (samples)"),
                            y=alt.Y("value:Q", title="Relative activity"),
                            color=alt.Color(
                                "channel:N",
                                title="Sensors",
                                scale=alt.Scale(
                                    range=["#4AA3A2", "#6CB0B0", "#8BC3C3", "#B0D8D8"]
                                ),
                            ),
                        )
                        .properties(
                            width="container",
                            height=220,
                            background="#F7FCFA",
                        )
                        .configure_axis(labelColor="#174C4F", titleColor="#174C4F")
                        .configure_legend(labelColor="#174C4F", titleColor="#174C4F")
                        .configure_view(strokeWidth=0)
                    )

                    st.altair_chart(line_chart, width="stretch")

                # --- AI Coach ---
                if ai_on:
                    if ai_on and not st.session_state.get("ollama_warmed"):
                        prewarm_ollama()
                    st.markdown(
                        """
                        <style>
                        .ai-mini {
                            background:#E8F7F3;
                            border:1px solid #CFE3E0;
                            border-radius:14px;
                            padding:.9rem 1.1rem;
                            color:#174C4F;
                            font-family:'Poppins',system-ui;
                            font-size:1rem; line-height:1.55;
                            margin-top:1.2rem;
                            max-width:900px; margin-left:auto; margin-right:auto;
                        }
                        .ai-mini-title { font-weight:600; margin-bottom:.4rem; color:#174C4F; }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.spinner("AI Coach is summarizing your session..."):
                        ai_text = ai_coach_quick_reflection(
                            calm_score,  # 0..1
                            st.session_state.get("participant_id"),
                            st.session_state.get("task"),
                            profile=st.session_state.get("profile"),
                        )
                        if ai_text:
                            st.markdown(
                                "<div class='ai-mini'><div class='ai-mini-title'>🧘 AI Coach Reflection</div>"
                                f"{ai_text}</div>",
                                unsafe_allow_html=True,
                            )

        elif mode == "Literal":
            calm_pct, focus_pct, stability, df_clean, to_plot = get_session_scores(df)

            if not to_plot:
                st.info("No numeric channels detected to interpret.")
            else:
                # For metaphors, keep the old “score” meaning: 0..1 stability
                score = stability

                # For stars, use calm percentage directly (0–100 → 0–1)
                score_for_stars = calm_pct / 100.0

                thresholds = [0.20, 0.40, 0.60, 0.80]
                stars = sum(score_for_stars >= t for t in thresholds)
                if score_for_stars >= thresholds[-1]:
                    stars = 5

                # --- Metaphors & Poems for each level ---
                METAPHORS = {
                    0: [
                        "Your session carried the rhythm of a storm — thoughts striking like lightning across restless waves. Each signal spoke of motion, not failure, but untamed energy looking for direction. Remember, even the wildest seas return to stillness once the winds grow tired.",
                        "Imagine your mind as a shaken snow globe — swirling, bright, beautiful chaos. Nothing wrong here; it only means there’s much energy waiting to settle. Let it fall grain by grain until the picture becomes clear again.",
                        "The current rushed fast today. Like a river after heavy rain, it didn’t know where to go first — everywhere, all at once. That too is part of flow. Calm will come as the water learns its bends.",
                        "Today felt like sailing through choppy waters — the waves were loud and close. A gentler pace and deeper breaths will calm the sea.",
                        "Your mind moved like branches in a sudden gust — lively, unsettled. A short pause could help the wind die down.",
                        "Stormy skies for now — flashes, rumbles, and quick shifts. Let’s wait a moment for the clouds to part."
                    ],
                    1: [
                        "The first quiet has begun — a whisper under the noise. Your signals show the moment before a wave smooths itself out, a glimpse of calm pressing through the surface. You are teaching your focus to breathe.",
                        "Today’s rhythm was tender, uncertain — like tuning a string that still hums between pitches. Not perfect, but true. The steadiness you seek is already humming underneath the rest.",
                        "You touched calm more than once, even if briefly. The mind wavered but never lost its course completely. Each return to breath was a small act of mastery.",
                        "Your focus flickered like a candle in a soft breeze. It wavered, found footing, and then danced again.",
                        "Imagine tuning an old radio: static giving way to a faint melody. You’re finding the signal — small adjustments, patient listening."
                    ],
                    2: [
                        "Your thoughts moved like a slow current — gentle surges, pauses, and quiet recovery. There is structure here, even if it feels soft. It’s the rhythm of learning balance, not yet still, but patient and aware.",
                        "The waves have softened. Ripples rise and fall, but between them — still water. You are learning the language of calm, one syllable at a time.",
                        "The signal breathes with you — imperfection in harmony. You are not forcing focus; you are finding it, moment by moment, breath by breath.",
                        "A river finding its groove — a few ripples, but the current is clear.",
                        "Clouds move, but light breaks through — you’re finding steadiness."
                    ],
                    3: [
                        "Your mind feels like a kite held by a steady hand — moving with the air but never out of reach. This is balance, not by control, but by trust in rhythm. The calm you practiced now carries you.",
                        "There was grace today in how you handled distraction — gentle correction instead of resistance. You are not fighting the wind anymore, you are gliding with it.",
                        "A quiet current flowed beneath the surface — signs of natural pacing, pauses that breathe, energy that sustains. The surface ripples, but the depth remains still.",
                        "A kite in smooth air — held, guided, and steady on the line."
                    ],
                    4: [
                        "Calm was your companion — a soft thread running through each second. The signal shows balance like a candle flame: alive, steady, aware. You found the place between effort and ease.",
                        "A still pond mirrored the sky in your focus — unbroken, reflecting everything yet disturbed by nothing. You’ve reached the rhythm where awareness becomes rest.",
                        "You are no longer searching for stillness — you are practicing it. The waves respond to you now, not the other way around.",
                        "A lantern glow — even and warm, guiding without flickering."
                    ],
                    5: [
                        "Pure coherence — your mind and breath moved as one. The signal is a quiet song, steady and unwavering. This is not just focus; it’s flow — effortless, grounded, luminous.",
                        "The session hums with symmetry — thought and silence alternating like day and night. You found the horizon line where calm meets clarity. 🌟",
                        "The mind became a mirror — reflecting, not reacting. Every motion knew its place. You didn’t chase calm; you became it.",
                        "A clear night sky — wide, quiet, and effortlessly steady. 🌟",
                        "Mountain air at dawn — still, bright, and unhurried.",
                        "Glass-calm water — reflections hold without a ripple."
                    ],
                }

                POEMS = {
                    0: [
                        "Winds crash and scatter,\nthoughts like leaves in stormlight —\nno rest,\nyet promise of calm.",
                        "The ocean shouted today,\nwhitecaps on every wave —\nremember:\nall storms fade."
                    ],
                    1: [
                        "Between waves of noise,\na rhythm starts to whisper —\nthe sea remembers peace.",
                        "Your mind hums like wings in air,\nfinding the beat,\nlosing it,\nfinding again."
                    ],
                    2: [
                        "A small fire steadying —\nits flame still sways,\nbut the warmth holds true.",
                        "The river hesitates,\nthen smooths —\nits song quieter,\nits path more sure."
                    ],
                    3: [
                        "Calm returns like tide —\nflowing without demand;\na balance learned,\nnot forced.",
                        "You stood in soft wind —\nthe silence between thoughts\nbecoming your anchor."
                    ],
                    4: [
                        "Quiet light pours inward —\nthe surface still,\nthe depths alive with knowing.",
                        "No waves remain —\nonly reflection:\nthe mind,\na mirror of sky."
                    ],
                    5: [
                        "The sea no longer moves,\nyet it lives in every wave.\nYour mind rests there —\nawake, vast, unbroken.",
                        "The breath and the thought move as one —\nno longer something to control,\njust something to notice.\nThis is the rhythm of calm."
                    ],
                }

                # Pick combined pool of metaphor + poem
                pool = METAPHORS.get(stars, []) + POEMS.get(stars, [])
                if not pool:
                    pool = METAPHORS.get(2, []) + POEMS.get(2, [])

                text = random.choice(pool)

                # --- technical section ---
                summary = st.session_state.get("summary", {}) or {}

                # --- Render torn-paper metaphor card ---
                calm_val = summary.get("calm_score", None)

                if calm_val is not None:
                    calm_stars = int(round(calm_val / 20))  # 0–100 mapped to 0–5 stars
                    calm_stars = max(0, min(calm_stars, 5))
                    stars_display = "★" * calm_stars + "☆" * (5 - calm_stars)

                    st.markdown(
                        f"""
                        <div class='lit-card'>
                            <div class='lit-body lit-quote'>{text}</div>
                            <div class='lit-foot'>(calm: {calm_val:.0f}% • {stars_display})</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Fallback to old behavior if calm_val is missing
                    st.markdown(
                        f"""
                        <div class='lit-card'>
                            <div class='lit-body lit-quote'>{text}</div>
                            <div class='lit-foot'>(stability score: {score:.2f} • {stars}★)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                task_label = (
                    summary.get("task_label")
                    or st.session_state.get("task")
                    or "your session"
                )

                # pull feature values from summary with safe fallbacks
                calm_val        = summary.get("calm_score", score)          # 0–100
                drift_val       = summary.get("drift_score")                # 0–100 (how much it kept changing)
                focus_val       = summary.get("focus_score")                # 0–100 (sustained task focus)
                movement_score  = summary.get("movement_score")             # 0–100
                calm_pct        = summary.get("percent_calm")               # 0–100 (% of time calm)
                movement_level  = summary.get("movement_level") or summary.get("movement")
                dom_band        = summary.get("dominant_band")
                session_len     = summary.get("duration_sec")


                # simple meaning helpers for bands
                BAND_MEANINGS = {
                    "delta": "very slow waves often linked with deep rest, drowsiness, or very low arousal.",
                    "theta": "slow, drifting waves often seen in light drowsiness, daydreaming, or internal focus.",
                    "alpha": "calm but awake rhythms, often strongest when you’re relaxed with eyes closed or gently focused.",
                    "beta":  "faster, more ‘busy’ waves that usually appear during active thinking, problem solving, or mental effort.",
                    "gamma": "very fast activity that can relate to high-level processing or complex tasks in some contexts.",
                }

                bullets = []

                # --- Calm / stability ---
                if calm_val is not None:
                    if calm_val >= 70:
                        quality = "mostly smooth and even"
                        tone = "Your brain kept a fairly steady rhythm instead of constantly spiking up and down."
                    elif calm_val >= 40:
                        quality = "moderately stable"
                        tone = "Your signal shows both calmer stretches and choppier parts, as if focus came in waves."
                    else:
                        quality = "quite changeable"
                        tone = "The signal jumped around more, which often happens when the mind is busy, distracted, or restless."

                    bullets.append(
                        f"Calm (stability): {calm_val:.0f}/100 – overall, your brainwaves were {quality}. "
                        f"{tone}"
                    )

                # --- Focus quality (beta vs. alpha/theta balance) ---
                if focus_val is not None:
                    if focus_val >= 70:
                        focus_text = (
                            "Your EEG shows a strong, task-aligned focus pattern — like your attention stayed "
                            "locked on the activity for most of the session."
                        )
                    elif focus_val >= 40:
                        focus_text = (
                            "There were stretches of good focus mixed with more diffuse or relaxed moments — "
                            "a natural back-and-forth between effort and letting the mind wander."
                        )
                    else:
                        focus_text = (
                            "The pattern leans more toward relaxed or drifting attention than sustained task focus, "
                            "which is common when the mind is tired, bored, or split between several thoughts."
                        )

                    bullets.append(
                        f"Focus quality: {focus_val:.0f}/100 – this reflects how strongly your brain activity "
                        f"lined up with focused mental effort versus a more diffuse or relaxed state. {focus_text}"
                    )

                # --- Focus / drift (how much things kept changing) ---
                if drift_val is not None:
                    if drift_val >= 70:
                        drift_text = (
                            "There were frequent shifts in the pattern, like your attention "
                            "kept hopping between focus, distraction, and re-focusing."
                        )
                    elif drift_val >= 40:
                        drift_text = (
                            "You show a mix of steady focus and natural wandering – some parts look locked in, "
                            "others more exploratory or distracted."
                        )
                    else:
                        drift_text = (
                            "The pattern changed more gently over time, which often matches sustained focus "
                            "or a stable mental state."
                        )

                    bullets.append(
                        f"Focus shifts (drift): {drift_val:.0f}/100 – this describes how much the signal "
                        f"rose and fell over time. {drift_text}"
                    )

                # --- Movement ---
                if movement_score is not None:
                    if movement_score >= 70:
                        move_text = (
                            "Very little muscle or movement noise – the headset likely sat still and your face "
                            "and jaw were relatively relaxed."
                        )
                    elif movement_score >= 40:
                        move_text = (
                            "Some movement or muscle activity is visible, but much of the recording still looks usable."
                        )
                    else:
                        move_text = (
                            "A lot of the signal is influenced by muscle tension, blinking, or head movement, "
                            "so some parts are harder to interpret cleanly."
                        )

                    bullets.append(
                        f"Body stillness (movement): {movement_score:.0f}/100 – higher values mean a cleaner, "
                        f"less ‘shaky’ signal. {move_text}"
                    )

                # --- Time in calm pattern ---
                if calm_pct is not None:
                    bullets.append(
                        f"Time in a calm pattern: ~{calm_pct:.0f}% of the recording showed a more settled rhythm, "
                        f"where the waves stayed relatively even instead of constantly spiking."
                    )

                # --- Qualitative movement label ---
                if movement_level is not None:
                    mv = str(movement_level).lower()
                    if mv in ("high", "medium"):
                        desc = "the raw data was noticeably shaped by motion or muscle tension."
                    else:
                        desc = "the raw data looked mostly clean, with only light movement artifacts."
                    bullets.append(
                        f"Overall movement level: {movement_level} – this means {desc}"
                    )

                # --- Dominant band explanation ---
                if dom_band:
                    # normalize band key: e.g. "alpha (8–12 Hz)" -> "alpha"
                    band_key = str(dom_band).split()[0].lower()
                    band_desc = BAND_MEANINGS.get(
                        band_key,
                        "a characteristic rhythm that reflects your dominant state during the session."
                    )

                    bullets.append(
                        f"Dominant frequency band: {dom_band} – this was the strongest rhythm in your EEG. "
                        f"In simple terms, that usually reflects {band_desc}"
                    )

                # --- Session length ---
                if session_len:
                    bullets.append(
                        f"Session length: about {int(round(session_len/60))} minute(s) of recorded activity."
                    )

                if not bullets:
                    bullets.append(
                        "Your EEG data was usable, but not detailed enough for a deeper breakdown."
                    )

                bullet_html = "".join(f"<li>{b}</li>" for b in bullets)

                st.markdown(
                    f"""
                    <div class="tech-card">
                        <div class="tech-title">Let’s get technical</div>
                        <div class="tech-body">
                            <p>Here’s what stood out in your <b>{task_label}</b> session based on your EEG signal:</p>
                            <ul>
                                {bullet_html}
                            </ul>
                            <p style="margin-top:.35rem; opacity:.8;">
                                Altogether, these results show how your brain behaved during the session. The stability score
                                reflects how smooth or irregular your activity was — steadier waves often appear when the mind
                                is calm or steadily focused, while larger fluctuations suggest mental effort, shifting attention,
                                or restlessness. The drift score tells us how often your focus changed direction. Movement values
                                show how much facial tension or body movement shaped the recording. And the dominant frequency band
                                highlights the general mode your brain spent the most time in (relaxed, alert, internally focused,
                                or drowsy). Combined, these features give a clear picture of the mental state you occupied while
                                doing the task.
                            </p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # --- AI coach ---
                if ai_on:
                    if ai_on and not st.session_state.get("ollama_warmed"):
                        prewarm_ollama()

                    with st.spinner("AI Coach is summarizing your session..."):
                        ai_text = ai_coach_quick_reflection(
                            score,
                            st.session_state.get("participant_id"),
                            st.session_state.get("task"),
                            profile=st.session_state.get("profile"),
                        )
                        if ai_text:
                            st.markdown(
                                "<div class='ai-mini'><div class='ai-mini-title'>🧘 AI Coach Reflection</div>"
                                f"{ai_text}</div>",
                                unsafe_allow_html=True,
                            )

        elif mode == "LLM":
            run_ai = False

            # If user clicked "Refresh": request exactly ONE fresh run
            if st.session_state.get("ai_soft_reset"):
                st.session_state["ai_soft_reset"] = False   
                run_ai = True
                # clear previous answer so we don't accidentally reuse it
                st.session_state.pop("ollama_last_response", None)
            # First time entering this mode: no cached answer yet → run once
            elif "ollama_last_response" not in st.session_state:
                run_ai = True

            st.markdown("### 🤖 AI-Coach Insights")
            if ai_on and not st.session_state.get("ollama_warmed"):
                prewarm_ollama()

            # Early "Start over" that preempts any streaming
            pre_l, pre_sp, pre_r = st.columns([6, 2, 2])
            with pre_r:
                if st.button("↺ Start over", key="early_start_over_llm"):
                    reset_session_for_new_run()
                    st.session_state["page"] = "Start"
                    st.rerun()
                    st.stop()   

            # ---- UI styling ----
            st.markdown("""
                <style>
                /* Keep spinner readable */
                [data-testid="stSpinner"] > div,
                .stSpinner, .stSpinner > div {
                    color: #174C4F !important;
                    opacity: 1 !important;
                    font-family:'Poppins',system-ui;
                    font-size: 1rem;
                }
                [data-testid="stSpinner"] svg { stroke: #4AA3A2 !important; }

                /* LLM coach layout: robot + speech bubble */
                .llm-coach {
                    display: flex;
                    align-items: flex-start;
                    gap: 0.75rem;
                    max-width: 900px;
                    margin: 0.9rem auto 0.4rem auto;
                }

                .llm-avatar {
                    width: 40px;
                    height: 40px;
                    border-radius: 999px;
                    background: #4AA3A2;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 6px 14px rgba(0,0,0,.18);
                    flex-shrink: 0;
                    font-size: 1.4rem;
                }

                .llm-avatar span {
                    display: block;
                    transform: translateY(1px);
                }

                .llm-bubble {
                    position: relative;
                    background:#E8F7F3;
                    border-radius: 18px;
                    border-top-left-radius: 6px;
                    padding: 0.9rem 1.1rem;
                    color:#174C4F !important;
                    font-family:'Poppins',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
                    line-height:1.55;
                    font-size:1.0rem;
                    box-shadow:0 4px 12px rgba(0,0,0,.06);
                    width: 100%;
                }

                /* little speech tail */
                .llm-bubble::before {
                    content: "";
                    position: absolute;
                    left: -7px;
                    top: 16px;
                    width: 14px;
                    height: 14px;
                    background: #E8F7F3;
                    border-left: 1px solid #CFE3E0;
                    border-bottom: 1px solid #CFE3E0;
                    transform: rotate(45deg);
                    box-shadow: -2px 2px 3px rgba(0,0,0,.03);
                }

                .llm-bubble *{
                    color:#174C4F !important;
                    opacity:1 !important;
                }
                </style>
            """, unsafe_allow_html=True)

            # ---- personalization + summary info ----
            participant = st.session_state.get("participant_id", "User")
            task = st.session_state.get("task", "focus training")
            summary_data = st.session_state.get("summary") or {}

            # session-level features from enhanced_summary
            calm_val        = summary_data.get("calm_score")        # 0–100, higher = smoother
            drift_val       = summary_data.get("drift_score")       # 0–100, higher = more shifts
            focus_val       = summary_data.get("focus_score")       # 0–100, higher = more sustained focus
            movement_score  = summary_data.get("movement_score")    # 0–100, higher = less motion noise
            movement_lvl    = summary_data.get("movement_level") or summary_data.get("movement")
            dom_band        = summary_data.get("dominant_band")
            session_len     = summary_data.get("duration_sec")
            task_label      = summary_data.get("task_label") or task or "this session"

            # personalization bits from profile
            prof = st.session_state.get("profile", {})
            personal_bits = []
            if prof.get("consent"):
                if prof.get("primary_goal"):
                    personal_bits.append(f"Goal: {prof['primary_goal']}.")
                if prof.get("eeg_experience"):
                    personal_bits.append(f"EEG experience: {prof['eeg_experience']}.")
                if prof.get("age_range"):
                    personal_bits.append(f"Age range: {prof['age_range']}.")
                if prof.get("gender"):
                    personal_bits.append(f"Gender: {prof['gender']}.")
                if prof.get("occupation"):
                    personal_bits.append(f"Occupation: {prof['occupation']}.")
            personal_str = " ".join(personal_bits) if personal_bits else "not provided."

            # compact, structured summary of the EEG metrics
            fact_lines = []

            if calm_val is not None:
                if calm_val < 30:
                    tone = "quite changeable and restless overall."
                elif calm_val < 60:
                    tone = "a mix of steady moments and clear fluctuations."
                else:
                    tone = "mostly smooth and even over time."
                fact_lines.append(
                    f"- Calm / stability score: about {calm_val:.0f}/100 – your brainwaves were {tone}"
                )

            if focus_val is not None:
                if focus_val < 30:
                    focus_tone = "focus was hard to hold for long; your mind likely wandered often."
                elif focus_val < 60:
                    focus_tone = "focus came and went, with periods of attention and periods of drift."
                else:
                    focus_tone = "focus was generally well maintained on the task."
                fact_lines.append(
                    f"- Focus quality score: about {focus_val:.0f}/100 – {focus_tone}"
                )

            if drift_val is not None:
                fact_lines.append(
                    f"- Attention drift score: about {drift_val:.0f}/100 "
                    f"(higher = more frequent changes in the pattern of activity)."
                )

            if movement_score is not None:
                fact_lines.append(
                    f"- Body stillness score: about {movement_score:.0f}/100 "
                    f"(higher = less muscle and movement noise in the signal)."
                )

            if movement_lvl is not None:
                mv = str(movement_lvl).lower()
                if mv in ("high", "medium"):
                    mv_desc = "the signal was noticeably affected by movement or muscle tension."
                else:
                    mv_desc = "the signal looked relatively clean, with little movement interference."
                fact_lines.append(
                    f"- Overall movement level: {movement_lvl} – {mv_desc}"
                )

            if dom_band:
                fact_lines.append(
                    f"- Dominant frequency band: {dom_band} "
                    f"(typical indicator of the prevailing mental state – e.g. relaxed, alert, drowsy)."
                )

            if session_len:
                minutes = int(round(session_len / 60))
                fact_lines.append(
                    f"- Session length: about {minutes} minute(s) of recorded activity."
                )

            if not fact_lines:
                fact_lines.append(
                    "- No detailed metrics were available; treat this as a very high-level impression."
                )

            session_facts = "\n".join(fact_lines)

            # ---- AI prompt ----
            SYSTEM = (
                "You are a calm, encouraging mindfulness and focus coach. "
                "You see structured EEG session metrics and personal context (goal, age range, gender, occupation, EEG experience). "
                "Give feedback that is specific to these metrics and this personal context, but explained in everyday language. "
                "Be honest: if stability or focus scores are low, clearly say that the brainwaves were restless or changeable; "
                "do not describe them as calm or smooth. If scores are high, you can highlight that as a strength. "
                "Avoid technical jargon like 'band power' or 'standard deviation'; instead say things like "
                "'your brainwaves looked smooth most of the time' or 'there were many small fluctuations'. "
                "Use short paragraphs and clear tips related to posture, breath, attention habits, or daily routines and anything else neccessary. "
                "Use the personal information to make suggestions feel more relevant, but never stereotype or make assumptions."
            )

            USER = f"""
            Participant ID: {participant}
            Task during this session: {task_label}

            EEG-based metrics:
            {session_facts}

            Personal context (only use if helpful and respectful):
            {personal_str}

            Using this information, write 2 short, warm paragraphs:
            - First, describe what seemed to go well and where the signal shows more fluctuation
              (for example in focus, calmness, or due to movement).
            - Second, give 2–3 concrete, practical suggestions they can try next time,
              tailored to their task and to their personal info (goal, age range, gender, occupation, EEG experience) whenever it helps.

            Speak directly to them in the second person ("you"), and end on an encouraging note.
            """.strip()

            FINAL_PROMPT = f"{SYSTEM}\n\nUser:\n{USER}\n\nCoach:"

            # ---- helper to finish if the model got cut off ----
            def finish_if_cut(text: str) -> str:
                """If the last char isn't terminal punctuation, ask the model to close cleanly."""
                t = (text or "").rstrip()
                if not t or t[-1] in ".!?":
                    return text
                try:
                    r = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": FAST_OLLAMA_MODEL,
                            "stream": False,
                            "keep_alive": "30m",
                            "prompt": f"{SYSTEM}\n\nUser: {USER}\nCoach: {text}\n\nFinish the last thought in one short sentence.",
                            "options": {
                                "temperature": 0.4,
                                "num_predict": 80,
                                "num_thread": NUM_THREADS,
                            },
                        },
                        timeout=60,
                    )
                    if r.status_code == 200:
                        extra = (r.json() or {}).get("response", "")
                        return text + (extra or "")
                except Exception:
                    pass
                return text      

            # ---- run model ----
            if not ollama_ready():
                st.error("Ollama is not reachable at http://localhost:11434 — open the app or start Ollama first.")
            else:
                status = st.empty()
                out = st.empty()

                if run_ai:
                    status.markdown(
                        """
                        <div class="llm-coach">
                            <div class="llm-avatar"><span>🤖</span></div>
                            <div class="llm-bubble">Analyzing your session…</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    collected = ""
                    buffer = ""
                    last_push = time.time()
                    CHARS_PER_PUSH = 120
                    SECS_PER_PUSH  = 0.12

                    try:
                        for piece in call_ollama_stream_buffered(
                            FINAL_PROMPT,
                            model=FAST_OLLAMA_MODEL,
                            temperature=0.4,
                            num_predict=420,
                            timeout=180,
                        ):
                            if not piece:
                                continue
                            collected += piece
                            buffer += piece
                            now = time.time()

                            # push updates occasionally
                            if len(buffer) >= CHARS_PER_PUSH or (now - last_push) >= SECS_PER_PUSH:
                                out.markdown(
                                    f"""
                                    <div class="llm-coach">
                                        <div class="llm-avatar"><span>🤖</span></div>
                                        <div class="llm-bubble">{collected}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                buffer = ""
                                last_push = now

                        # auto-finish if cut
                        if buffer or collected:
                            collected = finish_if_cut(collected)
                            out.markdown(
                                f"""
                                <div class="llm-coach">
                                    <div class="llm-avatar"><span>🤖</span></div>
                                    <div class="llm-bubble">{collected}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        status.empty()

                        # store final answer
                        st.session_state["ollama_last_response"] = collected

                        if not collected.strip():
                            st.warning("The AI Coach didn’t return any text. Try again in a moment.")

                    except ReadTimeout:
                        status.empty()
                        st.error("The AI Coach took too long to respond. Please try again.")
                    except ConnectionError:
                        status.empty()
                        st.error("Couldn’t connect to Ollama. Is it running?")
                    except Exception as e:
                        status.empty()
                        # if we have partial text, still show it and cache it
                        if collected.strip():
                            collected = finish_if_cut(collected)
                            out.markdown(
                                f"""
                                <div class="llm-coach">
                                    <div class="llm-avatar"><span>🤖</span></div>
                                    <div class="llm-bubble">{collected}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.session_state["ollama_last_response"] = collected
                        st.error(f"AI Coach error: {e}")

                # If we are NOT running the model this rerun, just reuse cached text
                else:
                    cached = st.session_state.get("ollama_last_response", "").strip()
                    if cached:
                        out.markdown(
                            f"""
                            <div class="llm-coach">
                                <div class="llm-avatar"><span>🤖</span></div>
                                <div class="llm-bubble">{cached}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # Should only happen if something went wrong and no answer was cached
                        st.info("Press “↺ Refresh” to generate AI feedback for this session.")

            # Soft reset button
            st.markdown("""
                <style>
                .display-actions { margin-top: 1.2rem; }
                .display-actions .hint { color:#174C4F; opacity:.75; font-size:.92rem; text-align:center; margin-bottom:.4rem; }
                div.stButton > button[kind="secondary"]{
                    background:#EAF4F2 !important; color:#174C4F !important; border:1px solid #CFE3E0 !important;
                    border-radius:14px !important; padding:.6rem 1.2rem !important; font-weight:600 !important;
                }
                div.stButton > button[kind="secondary"]:hover{
                    background:#D6EFEF !important;
                }
                </style>
            """, unsafe_allow_html=True)

    # center button
    _, center_btn, _ = st.columns([3, 2, 3])
    with center_btn:
        # In AI (LLM) mode: soft reset – just regenerate the AI feedback
        if mode == "LLM":
            if st.button("↺ Refresh", key="back_to_start_from_display", type="primary", use_container_width=True):
                st.session_state["ai_soft_reset"] = True

                for k in ["ollama_stream_buffer", "ollama_last_response", "ai_error"]:
                    st.session_state.pop(k, None)

                st.rerun()

        # In all other modes: hard reset – go back to Start and clear everything
        else:
            if st.button("↺ Start over", key="back_to_start_from_display", type="primary", use_container_width=True):
                reset_session_for_new_run()
                st.session_state["page"] = "Start"
                st.rerun()




