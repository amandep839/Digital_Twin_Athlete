import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
import io
import random

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Running Zone Classifier",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background: #0a0a0f; }
.block-container { padding: 2rem 2rem 2rem 2rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a0f2e 50%, #0f1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    margin: 0;
    font-weight: 300;
}
.hero-accent { color: #818cf8; }

/* Step cards */
.step-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
}
.step-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    color: #818cf8;
    background: rgba(129,140,248,0.1);
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 6px;
    padding: 4px 10px;
    letter-spacing: 0.1em;
}
.step-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0;
}

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 1rem 0;
}
.metric-card {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #818cf8;
    margin: 0;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #475569;
    margin: 6px 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Zone result badge */
.zone-result {
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin: 1rem 0;
    border: 2px solid;
}
.zone-name {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    line-height: 1;
}
.zone-conf {
    font-size: 1rem;
    margin: 0 0 0.5rem 0;
    opacity: 0.8;
}
.zone-pace {
    font-size: 0.9rem;
    opacity: 0.7;
    margin: 0;
}

/* Zone colors */
.zone-zone2   { background: rgba(59,130,246,0.1);  border-color: #3b82f6; color: #93c5fd; }
.zone-aerobic { background: rgba(34,197,94,0.1);   border-color: #22c55e; color: #86efac; }
.zone-tempo   { background: rgba(234,179,8,0.1);   border-color: #eab308; color: #fde047; }
.zone-intense { background: rgba(239,68,68,0.1);   border-color: #ef4444; color: #fca5a5; }

/* Progress steps in sidebar */
.nav-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 4px;
    cursor: pointer;
    transition: background 0.2s;
}
.nav-step.active { background: rgba(129,140,248,0.15); }
.nav-step.done   { opacity: 0.6; }
.nav-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #1e1e2e;
    border: 2px solid #2a2a4a;
    flex-shrink: 0;
}
.nav-dot.active { background: #818cf8; border-color: #818cf8; }
.nav-dot.done   { background: #22c55e; border-color: #22c55e; }
.nav-label { font-size: 0.85rem; color: #64748b; }
.nav-label.active { color: #c7d2fe; font-weight: 500; }

/* Table styling */
.dataframe { font-size: 0.8rem !important; }

/* Accuracy big number */
.acc-big {
    font-family: 'Space Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    color: #22c55e;
    text-align: center;
    line-height: 1;
    margin: 1rem 0 0.5rem;
}
.acc-label {
    text-align: center;
    color: #475569;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Confusion matrix cell */
.cm-grid {
    display: grid;
    gap: 4px;
    margin: 1rem 0;
}

/* Info box */
.info-box {
    background: rgba(129,140,248,0.08);
    border: 1px solid rgba(129,140,248,0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.875rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* Feature slider label */
.feat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: 0.05em;
}

hr { border-color: #1e1e2e; margin: 1.5rem 0; }

/* Override streamlit defaults */
.stButton > button {
    background: #818cf8;
    color: #0a0a0f;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover { background: #6366f1; color: white; }

.stSlider > div > div { accent-color: #818cf8; }
.stSelectbox label { color: #94a3b8 !important; font-size: 0.8rem !important; }
.stNumberInput label { color: #94a3b8 !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'cm' not in st.session_state:
    st.session_state.cm = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

ZONE_COLORS = {
    'zone2': 'zone-zone2',
    'aerobic': 'zone-aerobic',
    'tempo': 'zone-tempo',
    'intense': 'zone-intense'
}
ZONE_PACE = {
    'zone2':   '7:00 – 9:00 min/km · easy jog',
    'aerobic': '5:30 – 7:00 min/km · moderate run',
    'tempo':   '4:30 – 5:30 min/km · comfortably hard',
    'intense': '3:45 – 4:30 min/km · maximum effort',
}

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <p style='font-family:Space Mono,monospace;font-size:0.7rem;
    color:#475569;letter-spacing:0.15em;margin-bottom:1.5rem'>
    RUNNING ZONE CLASSIFIER
    </p>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Upload CSV"),
        ("02", "Explore Data"),
        ("03", "Clean Data"),
        ("04", "Train Model"),
        ("05", "Evaluate"),
        ("06", "Live Inference"),
    ]

    for i, (num, label) in enumerate(steps, 1):
        status = 'done' if st.session_state.step > i else ('active' if st.session_state.step == i else '')
        dot_class = f'nav-dot {status}'
        label_class = f'nav-label {status}'
        icon = '✓' if status == 'done' else num
        if st.button(f"  {icon}  {label}", key=f"nav_{i}",
                     use_container_width=True):
            if i <= st.session_state.step:
                st.session_state.step = i

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:0.75rem;color:#334155;line-height:1.6'>
    IoT + AI/ML<br>Digital Twin Project<br>
    <span style='color:#4a5568'>ESP8266 + MPU6050</span>
    </p>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1 class="hero-title">Running Zone <span class="hero-accent">Classifier</span></h1>
  <p class="hero-sub">IoT · Digital Twin · ML Classification · ESP8266 + MPU6050</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FEATURE EXTRACTION FUNCTION
# ─────────────────────────────────────────────
def extract_features_df(df, window_size=20, step=10):
    X, y = [], []
    for label in df['label'].unique():
        subset = df[df['label'] == label].reset_index(drop=True)
        for start in range(0, len(subset) - window_size, step):
            window = subset.iloc[start:start + window_size]
            feats = []
            for col in ['aX','aY','aZ','gX','gY','gZ']:
                vals = window[col].values
                feats.append(np.mean(vals))
                feats.append(np.std(vals))
                feats.append(np.sqrt(np.mean(vals**2)))
                feats.append(np.max(np.abs(vals)))
                feats.append(np.mean(np.abs(np.diff(vals))))
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)

# ─────────────────────────────────────────────
# STEP 1 — UPLOAD CSV
# ─────────────────────────────────────────────
if st.session_state.step == 1:
    st.markdown("""
    <div class="step-card">
      <div class="step-header">
        <span class="step-num">STEP 01</span>
        <p class="step-title">Upload your IMU dataset</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Upload the <code>cleaned_dataset.csv</code> file you downloaded earlier.
    The file should have columns: <code>timestamp_ms, aX, aY, aZ, gX, gY, gZ, label</code><br>
    Labels should be: <code>zone2 · aerobic · tempo · intense</code>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV file here",
                                type=['csv'],
                                label_visibility="collapsed")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            # handle no header
            if df.columns[0] not in ['timestamp_ms','aX']:
                df = pd.read_csv(uploaded,
                    names=['timestamp_ms','aX','aY','aZ','gX','gY','gZ','label'])

            st.session_state.df = df
            st.success(f"File loaded — {len(df):,} rows")

            # preview
            st.markdown("**Preview (first 5 rows):**")
            st.dataframe(df.head(), use_container_width=True)

            # quick stats
            counts = df['label'].value_counts()
            cols = st.columns(len(counts))
            for i, (label, count) in enumerate(counts.items()):
                with cols[i]:
                    st.metric(label, f"{count:,} rows")

            if st.button("Continue to Data Exploration →"):
                st.session_state.step = 2
                st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ─────────────────────────────────────────────
# STEP 2 — EXPLORE DATA
# ─────────────────────────────────────────────
elif st.session_state.step == 2:
    df = st.session_state.df
    if df is None:
        st.warning("Please upload a CSV first.")
        st.session_state.step = 1
        st.rerun()

    st.markdown("""
    <div class="step-card">
      <div class="step-header">
        <span class="step-num">STEP 02</span>
        <p class="step-title">Explore your data</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total rows", f"{len(df):,}")
    with col2:
        st.metric("Classes", df['label'].nunique())
    with col3:
        st.metric("Features", 6)
    with col4:
        st.metric("Duration (est.)", f"{len(df)/5/60:.1f} min")

    st.markdown("<hr>", unsafe_allow_html=True)

    # per-label stats
    st.markdown("**Per-zone statistics — aZ (vertical acceleration):**")
    stats_rows = []
    for label in ['zone2','aerobic','tempo','intense']:
        sub = df[df['label']==label]
        if len(sub) == 0:
            continue
        aZ = sub['aZ'].values
        stats_rows.append({
            'Zone': label,
            'Rows': len(sub),
            'aZ mean': f"{np.mean(aZ):.3f}",
            'aZ RMS': f"{np.sqrt(np.mean(aZ**2)):.3f}",
            'aZ std': f"{np.std(aZ):.3f}",
            'aZ peak': f"{np.max(np.abs(aZ)):.3f}",
            'Gyro mean': f"{np.mean(np.sqrt(sub['gX']**2+sub['gY']**2+sub['gZ']**2)):.3f}",
        })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # axis selector
    st.markdown("**Visualise axis per zone:**")
    axis_sel = st.selectbox("Select axis", ['aX','aY','aZ','gX','gY','gZ'])
    label_sel = st.selectbox("Select zone", df['label'].unique())

    sub = df[df['label']==label_sel][axis_sel].values[:200]
    chart_df = pd.DataFrame({axis_sel: sub})
    st.line_chart(chart_df, height=200)

    if st.button("Continue to Data Cleaning →"):
        st.session_state.step = 3
        st.rerun()

# ─────────────────────────────────────────────
# STEP 3 — CLEAN DATA
# ─────────────────────────────────────────────
elif st.session_state.step == 3:
    df = st.session_state.df
    if df is None:
        st.warning("Please upload a CSV first.")
        st.session_state.step = 1
        st.rerun()

    st.markdown("""
    <div class="step-card">
      <div class="step-header">
        <span class="step-num">STEP 03</span>
        <p class="step-title">Clean data — remove pauses + add synthetic intense class</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Two cleaning steps:<br>
    <b>1. Remove pause windows</b> — detects gaps &gt;500ms (e.g. dog interruptions!) and removes 
    surrounding settling rows.<br>
    <b>2. Generate synthetic intense data</b> — extrapolated from tempo statistics since 
    intense run data was not collected.
    </div>
    """, unsafe_allow_html=True)

    pause_thresh = st.slider("Pause detection threshold (ms)", 300, 2000, 500, 100)
    settling = st.slider("Settling rows to remove after each pause", 3, 20, 10)
    intense_rows_n = st.slider("Synthetic intense rows to generate", 500, 2000, 979, 100)

    if st.button("Run Cleaning Pipeline"):
        with st.spinner("Cleaning..."):

            def clean_label(subset, label, thresh, settling_n):
                sub = subset.copy().reset_index(drop=True)
                timestamps = sub['timestamp_ms'].values
                intervals = np.diff(timestamps)
                pause_pos = np.where(intervals > thresh)[0]
                to_drop = set()
                for pos in pause_pos:
                    for j in range(max(0, pos-3), pos+1):
                        to_drop.add(j)
                    for j in range(pos+1, min(len(sub), pos+1+settling_n)):
                        to_drop.add(j)
                aZ = sub['aZ'].values
                aX = sub['aX'].values
                aY = sub['aY'].values
                for i in range(5, len(sub)-5):
                    w_aZ = aZ[i-5:i+5]
                    w_aX = aX[i-5:i+5]
                    w_aY = aY[i-5:i+5]
                    if np.std(w_aZ)+np.std(w_aX)+np.std(w_aY) < 1.5:
                        to_drop.add(i)
                cleaned = sub.drop(index=list(to_drop)).reset_index(drop=True)
                return cleaned, len(pause_pos), len(to_drop)

            clean_parts = []
            results = []
            for label in df['label'].unique():
                subset = df[df['label']==label]
                cleaned, pauses, removed = clean_label(subset, label, pause_thresh, settling)
                clean_parts.append(cleaned)
                results.append({
                    'Zone': label,
                    'Before': len(subset),
                    'Pauses found': pauses,
                    'Rows removed': removed,
                    'After': len(cleaned),
                    'Retained %': f"{len(cleaned)/len(subset)*100:.1f}%"
                })

            clean_df = pd.concat(clean_parts, ignore_index=True)

            # generate intense
            np.random.seed(42)
            intense = []
            for i in range(intense_rows_n):
                intense.append({
                    'timestamp_ms': 9000000 + i*200,
                    'aX': np.random.normal(-6.0, 13.5),
                    'aY': np.random.normal( 8.0, 12.0),
                    'aZ': np.random.normal( 7.0, 12.5),
                    'gX': np.random.normal( 0.0,  3.2),
                    'gY': np.random.normal(0.05,  3.0),
                    'gZ': np.random.normal( 0.0,  2.8),
                    'label': 'intense'
                })
            intense_df = pd.DataFrame(intense)
            clean_df = pd.concat([clean_df, intense_df], ignore_index=True)
            results.append({
                'Zone': 'intense (synthetic)',
                'Before': 0,
                'Pauses found': '-',
                'Rows removed': '-',
                'After': intense_rows_n,
                'Retained %': '100%'
            })

            st.session_state.df_clean = clean_df

        st.success("Cleaning complete!")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        st.markdown("**Final dataset:**")
        final_counts = clean_df['label'].value_counts()
        cols = st.columns(len(final_counts))
        for i, (label, count) in enumerate(final_counts.items()):
            with cols[i]:
                st.metric(label, f"{count:,}")

        if st.button("Continue to Model Training →"):
            st.session_state.step = 4
            st.rerun()

# ─────────────────────────────────────────────
# STEP 4 — TRAIN MODEL
# ─────────────────────────────────────────────
elif st.session_state.step == 4:
    df_clean = st.session_state.df_clean
    if df_clean is None:
        st.warning("Please complete data cleaning first.")
        st.session_state.step = 3
        st.rerun()

    st.markdown("""
    <div class="step-card">
      <div class="step-header">
        <span class="step-num">STEP 04</span>
        <p class="step-title">Train the Random Forest classifier</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Feature extraction:</b> Each 20-row window → 30 features (mean, std, RMS, peak, jerk 
    for each of 6 axes). The Random Forest learns to distinguish zones from these patterns.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.slider("Number of trees", 50, 300, 100, 50)
    with col2:
        window_size = st.slider("Window size (rows)", 10, 40, 20, 5)
    with col3:
        test_size = st.slider("Test split %", 10, 40, 20, 5)

    if st.button("Train Model"):
        with st.spinner("Extracting features and training..."):
            X, y = extract_features_df(df_clean, window_size=window_size, step=window_size//2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42, stratify=y)

            model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred,
                                  labels=['zone2','aerobic','tempo','intense'])

            st.session_state.model = model
            st.session_state.accuracy = acc
            st.session_state.report = report
            st.session_state.cm = cm
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

        st.success("Training complete!")

        st.markdown(f"""
        <div class="acc-big">{acc*100:.1f}%</div>
        <div class="acc-label">overall accuracy</div>
        """, unsafe_allow_html=True)

        st.markdown("<br>**Training summary:**", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Windows extracted", f"{len(X):,}")
        c2.metric("Train windows", f"{len(X_train):,}")
        c3.metric("Test windows", f"{len(X_test):,}")

        # download model
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        st.download_button(
            label="Download zone_classifier.pkl",
            data=buf,
            file_name="zone_classifier_cleaned.pkl",
            mime="application/octet-stream"
        )

        if st.button("Continue to Evaluation →"):
            st.session_state.step = 5
            st.rerun()

# ─────────────────────────────────────────────
# STEP 5 — EVALUATE
# ─────────────────────────────────────────────
elif st.session_state.step == 5:
    if st.session_state.model is None:
        st.warning("Please train the model first.")
        st.session_state.step = 4
        st.rerun()

    model = st.session_state.model
    report = st.session_state.report
    cm = st.session_state.cm
    acc = st.session_state.accuracy

    st.markdown("""
    <div class="step-card">
      <div class="step-header">
        <span class="step-num">STEP 05</span>
        <p class="step-title">Model evaluation</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="acc-big">{acc*100:.1f}%</div>
    <div class="acc-label">overall accuracy on held-out test set</div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # per-class metrics
    st.markdown("**Per-zone performance:**")
    labels_order = ['zone2','aerobic','tempo','intense']
    metric_cols = st.columns(4)
    for i, label in enumerate(labels_order):
        if label in report:
            r = report[label]
            with metric_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                  <p class="metric-val">{r['f1-score']*100:.0f}%</p>
                  <p class="metric-label">{label} F1</p>
                  <p style="font-size:0.7rem;color:#334155;margin:6px 0 0">
                  P: {r['precision']*100:.0f}%  R: {r['recall']*100:.0f}%
                  </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # confusion matrix
    st.markdown("**Confusion matrix:**")
    cm_df = pd.DataFrame(cm,
                         index=[f'True: {l}' for l in labels_order],
                         columns=[f'Pred: {l}' for l in labels_order])
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    Diagonal values = correct predictions. Off-diagonal = misclassifications.
    Tempo↔Intense confusion is expected since intense data is synthetic.
    </div>
    """, unsafe_allow_html=True)

    # feature importance
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Top 10 most important features:**")
    feat_names = []
    for col in ['aX','aY','aZ','gX','gY','gZ']:
        for stat in ['mean','std','rms','peak','jerk']:
            feat_names.append(f"{col}_{stat}")

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    imp_df = pd.DataFrame({
        'Feature': [feat_names[i] for i in top_idx],
        'Importance': [f"{importances[i]:.4f}" for i in top_idx]
    })
    st.dataframe(imp_df, use_container_width=True, hide_index=True)

    if st.button("Continue to Live Inference →"):
        st.session_state.step = 6
        st.rerun()

# ─────────────────────────────────────────────
# STEP 6 — LIVE INFERENCE DASHBOARD
# ─────────────────────────────────────────────
elif st.session_state.step == 6:
    if st.session_state.model is None:
        st.warning("Please train the model first.")
        st.session_state.step = 4
        st.rerun()

    model = st.session_state.model

    st.markdown("""
    <div class="step-card">
      <div class="step-header">
        <span class="step-num">STEP 06</span>
        <p class="step-title">Live inference — manual feature input</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Enter feature values manually to simulate a running window and get instant zone classification.
    You can use quick-fill presets for each zone or enter custom values.
    </div>
    """, unsafe_allow_html=True)

    # presets
    PRESETS = {
        'zone2 (easy jog)': {
            'aX_mean':-3.6,'aX_std':4.0,'aX_rms':5.4,'aX_peak':12.0,'aX_jerk':6.0,
            'aY_mean':3.0, 'aY_std':3.9,'aY_rms':4.9,'aY_peak':10.0,'aY_jerk':5.5,
            'aZ_mean':8.7, 'aZ_std':7.2,'aZ_rms':11.2,'aZ_peak':27.0,'aZ_jerk':11.6,
            'gX_mean':0.0, 'gX_std':1.4,'gX_rms':1.4,'gX_peak':3.0,'gX_jerk':1.8,
            'gY_mean':0.0, 'gY_std':0.6,'gY_rms':0.6,'gY_peak':1.5,'gY_jerk':0.8,
            'gZ_mean':0.0, 'gZ_std':1.1,'gZ_rms':1.1,'gZ_peak':2.5,'gZ_jerk':1.4,
        },
        'aerobic (moderate)': {
            'aX_mean':1.4, 'aX_std':4.0,'aX_rms':4.2,'aX_peak':11.0,'aX_jerk':5.5,
            'aY_mean':7.8, 'aY_std':9.2,'aY_rms':12.0,'aY_peak':34.0,'aY_jerk':11.0,
            'aZ_mean':5.7, 'aZ_std':6.8,'aZ_rms':8.9,'aZ_peak':25.0,'aZ_jerk':9.9,
            'gX_mean':0.03,'gX_std':2.0,'gX_rms':2.0,'gX_peak':5.0,'gX_jerk':2.5,
            'gY_mean':0.05,'gY_std':1.3,'gY_rms':1.3,'gY_peak':3.5,'gY_jerk':1.7,
            'gZ_mean':0.0, 'gZ_std':1.5,'gZ_rms':1.5,'gZ_peak':3.8,'gZ_jerk':2.0,
        },
        'tempo (hard)': {
            'aX_mean':-5.6,'aX_std':9.6,'aX_rms':11.1,'aX_peak':30.0,'aX_jerk':12.0,
            'aY_mean':5.6, 'aY_std':8.9,'aY_rms':10.5,'aY_peak':35.0,'aY_jerk':11.5,
            'aZ_mean':6.0, 'aZ_std':9.0,'aZ_rms':10.8,'aZ_peak':40.0,'aZ_jerk':11.8,
            'gX_mean':0.0, 'gX_std':2.1,'gX_rms':2.1,'gX_peak':6.0,'gX_jerk':2.8,
            'gY_mean':0.05,'gY_std':2.0,'gY_rms':2.0,'gY_peak':5.5,'gY_jerk':2.6,
            'gZ_mean':0.0, 'gZ_std':1.9,'gZ_rms':1.9,'gZ_peak':5.0,'gZ_jerk':2.5,
        },
        'intense (maximum)': {
            'aX_mean':-6.0,'aX_std':13.5,'aX_rms':14.8,'aX_peak':40.0,'aX_jerk':18.0,
            'aY_mean':8.0, 'aY_std':12.0,'aY_rms':14.4,'aY_peak':42.0,'aY_jerk':16.0,
            'aZ_mean':7.0, 'aZ_std':12.5,'aZ_rms':14.3,'aZ_peak':45.0,'aZ_jerk':17.0,
            'gX_mean':0.0, 'gX_std':3.2,'gX_rms':3.2,'gX_peak':8.0,'gX_jerk':4.0,
            'gY_mean':0.05,'gY_std':3.0,'gY_rms':3.0,'gY_peak':7.5,'gY_jerk':3.8,
            'gZ_mean':0.0, 'gZ_std':2.8,'gZ_rms':2.8,'gZ_peak':7.0,'gZ_jerk':3.5,
        },
    }

    preset = st.selectbox("Quick-fill preset", ['Custom'] + list(PRESETS.keys()))

    st.markdown("<hr>", unsafe_allow_html=True)

    # feature sliders
    def get_val(key, default):
        if preset != 'Custom' and key in PRESETS[preset]:
            return float(PRESETS[preset][key])
        return default

    feat_vals = {}

    tabs = st.tabs(["Accelerometer aX", "Accelerometer aY",
                    "Accelerometer aZ", "Gyroscope gX",
                    "Gyroscope gY", "Gyroscope gZ"])

    axes = ['aX','aY','aZ','gX','gY','gZ']
    ranges = {
        'aX':(-20,20),'aY':(-20,20),'aZ':(-20,20),
        'gX':(-5,5),  'gY':(-5,5),  'gZ':(-5,5)
    }
    peak_ranges = {
        'aX':(-50,50),'aY':(-50,50),'aZ':(-50,50),
        'gX':(-10,10),'gY':(-10,10),'gZ':(-10,10)
    }

    for tab, axis in zip(tabs, axes):
        with tab:
            c1, c2, c3, c4, c5 = st.columns(5)
            lo, hi = ranges[axis]
            plo, phi = peak_ranges[axis]
            with c1:
                feat_vals[f'{axis}_mean'] = st.number_input(
                    f'{axis} mean', lo, hi,
                    get_val(f'{axis}_mean', 0.0), 0.1,
                    key=f'{axis}_mean')
            with c2:
                feat_vals[f'{axis}_std'] = st.number_input(
                    f'{axis} std', 0.0, abs(hi)*2,
                    get_val(f'{axis}_std', 1.0), 0.1,
                    key=f'{axis}_std')
            with c3:
                feat_vals[f'{axis}_rms'] = st.number_input(
                    f'{axis} RMS', 0.0, abs(hi)*2,
                    get_val(f'{axis}_rms', 1.0), 0.1,
                    key=f'{axis}_rms')
            with c4:
                feat_vals[f'{axis}_peak'] = st.number_input(
                    f'{axis} peak', 0.0, abs(phi),
                    get_val(f'{axis}_peak', 5.0), 0.5,
                    key=f'{axis}_peak')
            with c5:
                feat_vals[f'{axis}_jerk'] = st.number_input(
                    f'{axis} jerk', 0.0, 30.0,
                    get_val(f'{axis}_jerk', 2.0), 0.1,
                    key=f'{axis}_jerk')

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("CLASSIFY THIS WINDOW", use_container_width=True):
        # build feature vector (same order as training)
        feature_vector = []
        for axis in ['aX','aY','aZ','gX','gY','gZ']:
            for stat in ['mean','std','rms','peak','jerk']:
                feature_vector.append(feat_vals[f'{axis}_{stat}'])

        X_input = np.array(feature_vector).reshape(1, -1)

        zone = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        classes = model.classes_
        confidence = max(proba) * 100

        zone_class = ZONE_COLORS.get(zone, '')

        st.markdown(f"""
        <div class="zone-result {zone_class}">
          <p class="zone-name">{zone.upper()}</p>
          <p class="zone-conf">Confidence: {confidence:.1f}%</p>
          <p class="zone-pace">{ZONE_PACE.get(zone, '')}</p>
        </div>
        """, unsafe_allow_html=True)

        # probability breakdown
        st.markdown("**Probability breakdown:**")
        prob_cols = st.columns(len(classes))
        for i, (cls, prob) in enumerate(zip(classes, proba)):
            with prob_cols[i]:
                st.metric(cls, f"{prob*100:.1f}%")

        # pace recommendation
        st.markdown("<hr>", unsafe_allow_html=True)
        target = st.selectbox("Your target zone:", 
                              ['zone2','aerobic','tempo','intense'],
                              key='target_zone')
        
        zone_order = ['zone2','aerobic','tempo','intense']
        current_idx = zone_order.index(zone)
        target_idx  = zone_order.index(target)

        if current_idx == target_idx:
            st.success(f"You are in your target zone ({target}). Hold this pace!")
        elif current_idx < target_idx:
            st.warning(f"You are below target. Speed up to reach {target}.")
        else:
            st.info(f"You are above target. Slow down to reach {target}.")