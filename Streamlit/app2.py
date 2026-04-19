import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import joblib
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Running Zone Classifier",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding: 2rem 2rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #0f0f1a; border-right: 1px solid #1e1e2e; }

.hero {
    background: linear-gradient(135deg,#0f0f1a 0%,#1a0f2e 50%,#0f1a2e 100%);
    border: 1px solid #2a2a4a; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-50%; right:-10%;
    width:400px; height:400px;
    background:radial-gradient(circle,rgba(99,102,241,0.15) 0%,transparent 70%);
    border-radius:50%;
}
.hero-title { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700;
    color:#e2e8f0; margin:0 0 0.4rem; line-height:1.2; }
.hero-sub { font-size:0.95rem; color:#64748b; margin:0; }
.hero-accent { color:#818cf8; }

.step-card { background:#0f0f1a; border:1px solid #1e1e2e;
    border-radius:12px; padding:1.25rem 1.5rem; margin-bottom:1.2rem; }
.step-num { font-family:'Space Mono',monospace; font-size:0.65rem; font-weight:700;
    color:#818cf8; background:rgba(129,140,248,0.1); border:1px solid rgba(129,140,248,0.3);
    border-radius:6px; padding:3px 9px; letter-spacing:0.1em; }
.step-title { font-size:1.05rem; font-weight:600; color:#e2e8f0;
    margin:8px 0 0; }

.metric-card { background:#13131f; border:1px solid #1e1e2e;
    border-radius:10px; padding:1rem; text-align:center; }
.metric-val { font-family:'Space Mono',monospace; font-size:1.5rem;
    font-weight:700; color:#818cf8; margin:0; line-height:1; }
.metric-lbl { font-size:0.7rem; color:#475569; margin:5px 0 0;
    text-transform:uppercase; letter-spacing:0.08em; }

.zone-result { border-radius:16px; padding:2rem; text-align:center;
    margin:1rem 0; border:2px solid; }
.zone-name { font-family:'Space Mono',monospace; font-size:2.8rem;
    font-weight:700; margin:0 0 0.4rem; line-height:1; }
.zone-conf { font-size:0.95rem; margin:0 0 0.3rem; opacity:.8; }
.zone-pace { font-size:0.85rem; opacity:.7; margin:0; }

.zone-zone2   { background:rgba(59,130,246,0.1);  border-color:#3b82f6; color:#93c5fd; }
.zone-aerobic { background:rgba(34,197,94,0.1);   border-color:#22c55e; color:#86efac; }
.zone-tempo   { background:rgba(234,179,8,0.1);   border-color:#eab308; color:#fde047; }
.zone-intense { background:rgba(239,68,68,0.1);   border-color:#ef4444; color:#fca5a5; }

.acc-big { font-family:'Space Mono',monospace; font-size:3.5rem;
    font-weight:700; color:#22c55e; text-align:center; line-height:1; margin:0.8rem 0 0.3rem; }
.acc-lbl { text-align:center; color:#475569; font-size:0.75rem;
    text-transform:uppercase; letter-spacing:0.1em; }

.info-box { background:rgba(129,140,248,0.08); border:1px solid rgba(129,140,248,0.2);
    border-radius:10px; padding:0.9rem 1.1rem; margin:0.8rem 0;
    font-size:0.85rem; color:#94a3b8; line-height:1.6; }

hr { border-color:#1e1e2e; margin:1.2rem 0; }

.stButton>button { background:#818cf8; color:#0a0a0f; border:none;
    border-radius:8px; font-weight:600; font-family:'DM Sans',sans-serif;
    padding:0.55rem 1.2rem; width:100%; }
.stButton>button:hover { background:#6366f1; color:white; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, val in [('step',1),('df',None),('model',None),
                 ('accuracy',None),('report',None),('cm',None),
                 ('feature_cols',None),('model_name','')]:
    if key not in st.session_state:
        st.session_state[key] = val

FEATURE_COLS = [
    'mean_aX','mean_aY','mean_aZ','mean_gX','mean_gY','mean_gZ',
    'std_aX', 'std_aY', 'std_aZ', 'std_gX', 'std_gY', 'std_gZ',
    'rms_aX', 'rms_aY', 'rms_aZ', 'rms_gX', 'rms_gY', 'rms_gZ',
    'sma_acc','sma_gyro'
]

ZONE_COLORS = {
    'zone2':'zone-zone2','aerobic':'zone-aerobic',
    'tempo':'zone-tempo','intense':'zone-intense'
}
ZONE_PACE = {
    'zone2':  '7:00–9:00 min/km · easy jog',
    'aerobic':'5:30–7:00 min/km · moderate run',
    'tempo':  '4:30–5:30 min/km · comfortably hard',
    'intense':'3:45–4:30 min/km · maximum effort',
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <p style='font-family:Space Mono,monospace;font-size:0.65rem;
    color:#475569;letter-spacing:0.15em;margin-bottom:1.2rem'>
    RUNNING ZONE CLASSIFIER
    </p>""", unsafe_allow_html=True)

    steps = [
        ("01","Upload Feature CSV"),
        ("02","Explore Data"),
        ("03","Train Model"),
        ("04","Evaluate"),
        ("05","Inference Dashboard"),
    ]
    for i,(num,label) in enumerate(steps,1):
        done  = st.session_state.step > i
        active= st.session_state.step == i
        icon  = "✓" if done else num
        color = "#22c55e" if done else ("#c7d2fe" if active else "#475569")
        st.markdown(f"""
        <div style='padding:8px 10px;border-radius:8px;margin-bottom:3px;
        background:{"rgba(129,140,248,0.12)" if active else "transparent"}'>
        <span style='font-family:Space Mono,monospace;font-size:0.7rem;
        color:{color}'>{icon}</span>
        <span style='font-size:0.82rem;color:{color};margin-left:8px'>{label}</span>
        </div>""", unsafe_allow_html=True)
        if i <= st.session_state.step:
            if st.button(f"Go to {label}", key=f"nav_{i}",
                         use_container_width=True):
                st.session_state.step = i
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<p style='font-size:0.72rem;color:#334155;line-height:1.7'>
    Features from MATLAB<br>Phase 1 extraction<br>
    <span style='color:#4a5568'>20 features · 4 zones</span></p>""",
    unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1 class="hero-title">Running Zone <span class="hero-accent">Classifier</span></h1>
  <p class="hero-sub">MATLAB Feature Extraction · Scikit-learn Training · Live Inference Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STEP 1 — UPLOAD
# ─────────────────────────────────────────────
if st.session_state.step == 1:
    st.markdown("""
    <div class="step-card">
      <span class="step-num">STEP 01</span>
      <p class="step-title">Upload Model_Ready_Features_Fixed.csv</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Upload the <b>Model_Ready_Features_Fixed.csv</b> file generated by your MATLAB Phase 1 script.
    It contains 20 pre-extracted features per window:<br>
    <code>mean · std · rms</code> for each of 6 axes + <code>sma_acc · sma_gyro</code>
    + a <code>label</code> column (zone2 / aerobic / tempo / intense).
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop CSV here", type=['csv'],
                                label_visibility="collapsed")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            # validate columns
            missing = [c for c in FEATURE_COLS + ['label'] if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.session_state.df = df
                st.success(f"Loaded — {len(df):,} windows · {df['label'].nunique()} classes")
                st.dataframe(df.head(5), use_container_width=True)

                counts = df['label'].value_counts()
                cols = st.columns(len(counts))
                for i,(lbl,cnt) in enumerate(counts.items()):
                    with cols[i]:
                        st.metric(lbl, f"{cnt} windows")

                if st.button("Continue to Data Exploration →"):
                    st.session_state.step = 2
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# STEP 2 — EXPLORE
# ─────────────────────────────────────────────
elif st.session_state.step == 2:
    df = st.session_state.df
    if df is None:
        st.session_state.step = 1; st.rerun()

    st.markdown("""
    <div class="step-card">
      <span class="step-num">STEP 02</span>
      <p class="step-title">Explore the feature dataset</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total windows", f"{len(df):,}")
    c2.metric("Features", 20)
    c3.metric("Classes", df['label'].nunique())
    c4.metric("Intense windows", f"{len(df[df.label=='intense'])}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Feature means per zone:**")

    stat_rows = []
    for lbl in ['zone2','aerobic','tempo','intense']:
        sub = df[df['label']==lbl]
        if len(sub)==0: continue
        stat_rows.append({
            'Zone': lbl,
            'Windows': len(sub),
            'rms_aZ': f"{sub.rms_aZ.mean():.3f}",
            'std_aZ': f"{sub.std_aZ.mean():.3f}",
            'sma_acc': f"{sub.sma_acc.mean():.3f}",
            'sma_gyro': f"{sub.sma_gyro.mean():.3f}",
            'mean_aY': f"{sub.mean_aY.mean():.3f}",
        })
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Visualise feature distribution:**")
    feat_sel = st.selectbox("Feature", FEATURE_COLS, index=14)  # default rms_aZ
    zone_sel = st.selectbox("Zone", ['all'] + list(df['label'].unique()))

    if zone_sel == 'all':
        chart_df = pd.DataFrame({
            lbl: df[df['label']==lbl][feat_sel].values[:100]
            for lbl in df['label'].unique() if lbl in df['label'].values
        })
    else:
        chart_df = pd.DataFrame({zone_sel: df[df['label']==zone_sel][feat_sel].values[:200]})
    st.line_chart(chart_df, height=220)

    st.markdown("""
    <div class="info-box">
    <b>Key observation:</b> sma_acc and rms_aZ both increase from zone2 → aerobic → tempo → intense,
    confirming the IMU captures running intensity correctly. The model will learn these boundaries.
    </div>""", unsafe_allow_html=True)

    if st.button("Continue to Model Training →"):
        st.session_state.step = 3; st.rerun()

# ─────────────────────────────────────────────
# STEP 3 — TRAIN
# ─────────────────────────────────────────────
elif st.session_state.step == 3:
    df = st.session_state.df
    if df is None:
        st.session_state.step = 1; st.rerun()

    st.markdown("""
    <div class="step-card">
      <span class="step-num">STEP 03</span>
      <p class="step-title">Train the classifier</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Features are already extracted by MATLAB — no windowing needed here.
    Each row = one 2-second window = one prediction. Just pick a model and train.
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        model_choice = st.selectbox("Algorithm", [
            "Random Forest",
            "Gradient Boosting",
            "SVM (RBF kernel)",
        ])
    with col2:
        test_pct = st.slider("Test split %", 10, 40, 20, 5)
    with col3:
        if model_choice == "Random Forest":
            n_est = st.slider("Number of trees", 50, 300, 100, 50)
        elif model_choice == "Gradient Boosting":
            n_est = st.slider("Estimators", 50, 200, 100, 50)
        else:
            n_est = 100  # unused for SVM

    X = df[FEATURE_COLS].values
    y = df['label'].values

    if st.button("Train Model"):
        with st.spinner("Training..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_pct/100, random_state=42, stratify=y)

            if model_choice == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=n_est, random_state=42, n_jobs=-1)
            elif model_choice == "Gradient Boosting":
                model = GradientBoostingClassifier(
                    n_estimators=n_est, random_state=42)
            else:
                from sklearn.svm import SVC
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
                ])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm     = confusion_matrix(y_test, y_pred,
                                      labels=['zone2','aerobic','tempo','intense'])

            # cross-val
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

            st.session_state.model      = model
            st.session_state.accuracy   = acc
            st.session_state.report     = report
            st.session_state.cm         = cm
            st.session_state.model_name = model_choice

        st.success("Training complete!")

        st.markdown(f"""
        <div class="acc-big">{acc*100:.1f}%</div>
        <div class="acc-lbl">test accuracy</div>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Train windows", len(X_train))
        c2.metric("Test windows",  len(X_test))
        c3.metric("CV mean acc",   f"{cv_scores.mean()*100:.1f}%")

        # download pkl
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        st.download_button(
            "Download zone_classifier.pkl",
            data=buf,
            file_name="zone_classifier.pkl",
            mime="application/octet-stream"
        )

        if st.button("Continue to Evaluation →"):
            st.session_state.step = 4; st.rerun()

# ─────────────────────────────────────────────
# STEP 4 — EVALUATE
# ─────────────────────────────────────────────
elif st.session_state.step == 4:
    if st.session_state.model is None:
        st.session_state.step = 3; st.rerun()

    model  = st.session_state.model
    report = st.session_state.report
    cm     = st.session_state.cm
    acc    = st.session_state.accuracy

    st.markdown("""
    <div class="step-card">
      <span class="step-num">STEP 04</span>
      <p class="step-title">Evaluate model performance</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="acc-big">{acc*100:.1f}%</div>
    <div class="acc-lbl">overall accuracy — {st.session_state.model_name}</div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # per-class F1
    st.markdown("**Per-zone F1 scores:**")
    labels_order = ['zone2','aerobic','tempo','intense']
    mc = st.columns(4)
    for i,lbl in enumerate(labels_order):
        if lbl in report:
            r = report[lbl]
            with mc[i]:
                st.markdown(f"""
                <div class="metric-card">
                  <p class="metric-val">{r['f1-score']*100:.0f}%</p>
                  <p class="metric-lbl">{lbl}</p>
                  <p style="font-size:0.7rem;color:#334155;margin:5px 0 0">
                  P:{r['precision']*100:.0f}%  R:{r['recall']*100:.0f}%  n:{int(r['support'])}
                  </p>
                </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # confusion matrix
    st.markdown("**Confusion matrix:**")
    cm_df = pd.DataFrame(
        cm,
        index  =[f"True: {l}" for l in labels_order],
        columns=[f"Pred: {l}" for l in labels_order]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    Diagonal = correct predictions. Any tempo↔intense confusion is expected
    since intense has only 52 real windows. Collecting a real intense run will
    improve this boundary significantly.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # feature importance (RF / GB only)
    if hasattr(model, 'feature_importances_'):
        st.markdown("**Top 10 most important features:**")
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:10]
        imp_df = pd.DataFrame({
            'Feature':    [FEATURE_COLS[i] for i in top_idx],
            'Importance': [f"{imp[i]:.4f}"  for i in top_idx],
        })
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    if st.button("Continue to Inference Dashboard →"):
        st.session_state.step = 5; st.rerun()

# ─────────────────────────────────────────────
# STEP 5 — INFERENCE DASHBOARD
# ─────────────────────────────────────────────
elif st.session_state.step == 5:
    if st.session_state.model is None:
        st.session_state.step = 3; st.rerun()

    model = st.session_state.model

    st.markdown("""
    <div class="step-card">
      <span class="step-num">STEP 05</span>
      <p class="step-title">Live inference dashboard — enter feature values manually</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Simulate a 2-second IMU window by entering the 20 feature values below.
    Use a <b>Quick Preset</b> to auto-fill realistic values for each zone,
    or enter your own. The model classifies instantly.
    </div>""", unsafe_allow_html=True)

    # ── presets derived from actual data stats ──
    PRESETS = {
        'zone2 (easy jog)': {
            'mean_aX':-3.6,'mean_aY':3.0, 'mean_aZ':8.7,
            'mean_gX':0.0, 'mean_gY':0.0, 'mean_gZ':0.0,
            'std_aX':4.0,  'std_aY':3.9,  'std_aZ':7.2,
            'std_gX':1.4,  'std_gY':0.6,  'std_gZ':1.1,
            'rms_aX':5.4,  'rms_aY':4.9,  'rms_aZ':11.2,
            'rms_gX':1.4,  'rms_gY':0.6,  'rms_gZ':1.1,
            'sma_acc':16.5,'sma_gyro':0.5,
        },
        'aerobic (moderate)': {
            'mean_aX':1.4, 'mean_aY':7.8, 'mean_aZ':5.7,
            'mean_gX':0.03,'mean_gY':0.05,'mean_gZ':0.0,
            'std_aX':4.0,  'std_aY':9.2,  'std_aZ':6.8,
            'std_gX':2.0,  'std_gY':1.3,  'std_gZ':1.5,
            'rms_aX':4.2,  'rms_aY':12.0, 'rms_aZ':8.6,
            'rms_gX':2.0,  'rms_gY':1.3,  'rms_gZ':1.5,
            'sma_acc':18.4,'sma_gyro':2.3,
        },
        'tempo (hard)': {
            'mean_aX':-5.6,'mean_aY':5.6, 'mean_aZ':6.0,
            'mean_gX':0.0, 'mean_gY':0.05,'mean_gZ':0.0,
            'std_aX':9.6,  'std_aY':8.9,  'std_aZ':9.0,
            'std_gX':2.1,  'std_gY':2.0,  'std_gZ':1.9,
            'rms_aX':11.1, 'rms_aY':10.5, 'rms_aZ':10.7,
            'rms_gX':2.1,  'rms_gY':2.0,  'rms_gZ':1.9,
            'sma_acc':22.8,'sma_gyro':2.9,
        },
        'intense (maximum)': {
            'mean_aX':-6.0,'mean_aY':8.0, 'mean_aZ':7.0,
            'mean_gX':0.0, 'mean_gY':0.05,'mean_gZ':0.0,
            'std_aX':13.5, 'std_aY':12.0, 'std_aZ':12.5,
            'std_gX':3.2,  'std_gY':3.0,  'std_gZ':2.8,
            'rms_aX':14.8, 'rms_aY':14.4, 'rms_aZ':14.1,
            'rms_gX':3.2,  'rms_gY':3.0,  'rms_gZ':2.8,
            'sma_acc':34.5,'sma_gyro':4.5,
        },
    }

    preset = st.selectbox("Quick-fill preset",
                          ['Custom'] + list(PRESETS.keys()))

    def pval(key, default):
        if preset != 'Custom' and key in PRESETS.get(preset,{}):
            return float(PRESETS[preset][key])
        return default

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── feature input in organised tabs ──
    feat_vals = {}
    tabs = st.tabs(["Mean values", "Std values", "RMS values", "SMA values"])

    with tabs[0]:
        st.markdown("*Mean acceleration and gyroscope per axis*")
        c1,c2,c3 = st.columns(3)
        c4,c5,c6 = st.columns(3)
        feat_vals['mean_aX'] = c1.number_input('mean_aX', -20.0,20.0, pval('mean_aX',0.0),0.1)
        feat_vals['mean_aY'] = c2.number_input('mean_aY', -20.0,20.0, pval('mean_aY',0.0),0.1)
        feat_vals['mean_aZ'] = c3.number_input('mean_aZ', -20.0,20.0, pval('mean_aZ',9.8),0.1)
        feat_vals['mean_gX'] = c4.number_input('mean_gX', -5.0, 5.0,  pval('mean_gX',0.0),0.01)
        feat_vals['mean_gY'] = c5.number_input('mean_gY', -5.0, 5.0,  pval('mean_gY',0.0),0.01)
        feat_vals['mean_gZ'] = c6.number_input('mean_gZ', -5.0, 5.0,  pval('mean_gZ',0.0),0.01)

    with tabs[1]:
        st.markdown("*Standard deviation — higher = more variable movement*")
        c1,c2,c3 = st.columns(3)
        c4,c5,c6 = st.columns(3)
        feat_vals['std_aX'] = c1.number_input('std_aX', 0.0,20.0, pval('std_aX',1.0),0.1)
        feat_vals['std_aY'] = c2.number_input('std_aY', 0.0,20.0, pval('std_aY',1.0),0.1)
        feat_vals['std_aZ'] = c3.number_input('std_aZ', 0.0,20.0, pval('std_aZ',1.0),0.1)
        feat_vals['std_gX'] = c4.number_input('std_gX', 0.0,10.0, pval('std_gX',0.5),0.1)
        feat_vals['std_gY'] = c5.number_input('std_gY', 0.0,10.0, pval('std_gY',0.5),0.1)
        feat_vals['std_gZ'] = c6.number_input('std_gZ', 0.0,10.0, pval('std_gZ',0.5),0.1)

    with tabs[2]:
        st.markdown("*RMS — root mean square, captures energy of movement*")
        c1,c2,c3 = st.columns(3)
        c4,c5,c6 = st.columns(3)
        feat_vals['rms_aX'] = c1.number_input('rms_aX', 0.0,25.0, pval('rms_aX',1.0),0.1)
        feat_vals['rms_aY'] = c2.number_input('rms_aY', 0.0,25.0, pval('rms_aY',1.0),0.1)
        feat_vals['rms_aZ'] = c3.number_input('rms_aZ', 0.0,25.0, pval('rms_aZ',9.8),0.1)
        feat_vals['rms_gX'] = c4.number_input('rms_gX', 0.0,10.0, pval('rms_gX',0.5),0.1)
        feat_vals['rms_gY'] = c5.number_input('rms_gY', 0.0,10.0, pval('rms_gY',0.5),0.1)
        feat_vals['rms_gZ'] = c6.number_input('rms_gZ', 0.0,10.0, pval('rms_gZ',0.5),0.1)

    with tabs[3]:
        st.markdown("*Signal Magnitude Area — overall activity intensity*")
        c1, c2 = st.columns(2)
        feat_vals['sma_acc']  = c1.number_input('sma_acc  (accel)',  0.0,60.0, pval('sma_acc',10.0),0.5)
        feat_vals['sma_gyro'] = c2.number_input('sma_gyro (gyro)',   0.0,15.0, pval('sma_gyro',0.5),0.1)

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("CLASSIFY THIS WINDOW", use_container_width=True):
        # build feature vector in exact FEATURE_COLS order
        fv = np.array([feat_vals[c] for c in FEATURE_COLS]).reshape(1,-1)

        zone  = model.predict(fv)[0]
        proba = model.predict_proba(fv)[0]
        conf  = max(proba)*100
        zc    = ZONE_COLORS.get(zone,'')

        st.markdown(f"""
        <div class="zone-result {zc}">
          <p class="zone-name">{zone.upper()}</p>
          <p class="zone-conf">Confidence: {conf:.1f}%</p>
          <p class="zone-pace">{ZONE_PACE.get(zone,'')}</p>
        </div>""", unsafe_allow_html=True)

        # probability bar
        st.markdown("**Class probabilities:**")
        pc = st.columns(len(model.classes_))
        for i,(cls,prob) in enumerate(zip(model.classes_, proba)):
            with pc[i]:
                st.metric(cls, f"{prob*100:.1f}%")

        st.markdown("<hr>", unsafe_allow_html=True)

        # pace coach
        st.markdown("**Pace coach:**")
        target = st.selectbox("Your target zone:",
                              ['zone2','aerobic','tempo','intense'],
                              key='tgt')
        z_order = ['zone2','aerobic','tempo','intense']
        ci = z_order.index(zone)
        ti = z_order.index(target)

        if ci == ti:
            st.success(f"You are exactly in your target zone ({target}). Hold this pace!")
        elif ci < ti:
            st.warning(f"Below target — you are in {zone}. Speed up to reach {target}.")
        else:
            st.info(f"Above target — you are in {zone}. Slow down to reach {target}.")

        # show feature vector as table
        with st.expander("Show full feature vector sent to model"):
            fv_df = pd.DataFrame([feat_vals])
            st.dataframe(fv_df, use_container_width=True)

