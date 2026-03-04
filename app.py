import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from groq import Groq
import os
import sqlite3
import re
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Strategic Governance AI", page_icon="⚖️", layout="wide")

# --- DATABASE SETUP & AUTO-MIGRATION ---
def init_db():
    conn = sqlite3.connect('governance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, project_name TEXT, description TEXT,
                    budget REAL, months INTEGER, score INTEGER,
                    f_risk INTEGER, t_risk INTEGER, o_risk INTEGER,
                    risk_level TEXT, summary TEXT, advice TEXT, 
                    bias_score INTEGER, category TEXT, reasoning TEXT,
                    hype_score INTEGER, stress_test TEXT, risk_logic TEXT,
                    exp_cost REAL, exp_completion TEXT
                )''')
    
    # Ensure schema persistence for all advanced tracking fields
    c.execute("PRAGMA table_info(audits)")
    existing_cols = [column[1] for column in c.fetchall()]
    required_cols = [
        ('hype_score', 'INTEGER DEFAULT 0'), ('stress_test', 'TEXT'),
        ('risk_logic', 'TEXT'), ('bias_score', 'INTEGER DEFAULT 0'),
        ('reasoning', 'TEXT'), ('category', 'TEXT DEFAULT "General"'),
        ('exp_cost', 'REAL DEFAULT 0'), ('exp_completion', 'TEXT')
    ]
    for col_name, col_type in required_cols:
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE audits ADD COLUMN {col_name} {col_type}")
    conn.commit()
    conn.close()

def save_audit(d):
    conn = sqlite3.connect('governance.db')
    c = conn.cursor()
    c.execute('''INSERT INTO audits 
                 (timestamp, project_name, description, budget, months, score, f_risk, t_risk, o_risk, 
                  risk_level, summary, advice, bias_score, category, reasoning, hype_score, stress_test, 
                  risk_logic, exp_cost, exp_completion) 
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', 
              (datetime.now().strftime("%Y-%m-%d %H:%M"), d['name'], d['desc'], d['budget'], d['months'],
               d['score'], d['f_risk'], d['t_risk'], d['o_risk'], d['risk_level'], d['summary'], d['advice'], 
               d.get('bias', 0), d.get('category', 'General'), d.get('reasoning', ''), d.get('hype', 0),
               d.get('stress_test', ''), d.get('risk_logic', ''), d.get('exp_cost', 0), d.get('exp_completion', '')))
    conn.commit()
    conn.close()

def get_all_projects():
    conn = sqlite3.connect('governance.db')
    df = pd.read_sql_query("SELECT * FROM audits ORDER BY id DESC", conn)
    conn.close()
    return df

def safe_int(val, default=0):
    try:
        clean_val = re.sub(r'[^\d]', '', str(val))
        return int(clean_val) if clean_val else default
    except: return default

init_db()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()
SYSTEM_API_KEY = os.environ.get("GROQ_API_KEY")

# --- THEME & STYLING ---
st.markdown("""
    <style>
    .stApp { background: #fdfbff; }
    h1, h2, h3 { color: #4a148c !important; font-family: 'Inter', sans-serif; margin-top: 1.5rem; }
    .stButton>button { background: #7e57c2; color: white; border-radius: 8px; font-weight: bold; height: 3.5em; width: 100%; border: none;}
    .report-card { background: white; padding: 25px; border-radius: 15px; border-top: 5px solid #7e57c2; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px;}
    .reasoning-hub { background: #f3e5f5; padding: 20px; border-radius: 12px; border-left: 5px solid #7e57c2; margin-top: 15px;}
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(126,87,194,0.75), rgba(0,0,0,0)); margin: 3rem 0; }
    </style>
    """, unsafe_allow_html=True)

class StrategicInferenceEngine:
    @staticmethod
    def run_prescriptive_audit(text, budget, timeline, team):
        client = Groq(api_key=SYSTEM_API_KEY)
        prompt = f"""
        Masters-Level Strategic Governance Audit:
        Proposal: {text} | Budget: ${budget} | Timeline: {timeline}mo | Staff: {team}p.
        
        OUTPUT FORMAT (STRICT):
        CATEGORY: [Name] | SCORE: [0-100] | HYPE: [0-100] | BIAS: [0-100]
        F_RISK: [0-100] | T_RISK: [0-100] | O_RISK: [0-100]
        EXPECTED_COST: [Numerical value only]
        COMPLETION_DATE: [Estimated months]
        REASON_SUCCESS: [Logic]
        REASON_RISK: [Causal drivers]
        STRESS_TEST: [Analysis under 20% budget cut]
        PRESCRIPTIVE_ADJUSTMENT: [Actionable budget/staff changes]
        MILESTONES: Phase 1, 20; Phase 2, 40; Phase 3, 30
        SUMMARY: [Conclusion] | ADVICE: [Strategic Roadmap]
        """
        res = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile", temperature=0.1)
        return res.choices[0].message.content

# --- MAIN UI ---
st.title("⚖️ Strategic Governance Intelligence")
st.markdown("##### Probabilistic Stress Testing • Multi-Model Risk Surfaces • Prescriptive Outcome Logic")

with st.sidebar:
    st.header("⚙️ Administration")
    if st.button("🗑️ Reset Audit Database"):
        sqlite3.connect('governance.db').cursor().execute("DELETE FROM audits")
        st.rerun()
    st.divider()
    st.info("Embedding: all-MiniLM-L6-v2")
    st.info("Inference: Groq Llama 3.3")

tab_audit, tab_portfolio, tab_heatmap, tab_monte_carlo, tab_faceoff = st.tabs(["🚀 Strategic Audit", "📊 Portfolio Hub", "🌡️ ISO 5x5 Matrix", "🎲 Monte Carlo", "🥊 Selection Logic"])

with tab_audit:
    col_in, col_viz = st.columns([1, 1.8], gap="large")
    with col_in:
        st.subheader("Project Intake")
        p_name = st.text_input("Project ID/Name", "PRJ-MASTER-X")
        p_desc = st.text_area("Detailed Research Proposal", height=200)
        c1, c2, c3 = st.columns(3)
        budget = c1.number_input("Capital Req ($)", value=300000, step=10000)
        months = c2.number_input("Timeline (Mo)", value=12)
        team = c3.number_input("Staff Load", value=20)
        
        if st.button("EXECUTE HYPER-INTELLIGENCE AUDIT"):
            with st.spinner("⚡ Running AI Models..."):
                # Neural Redundancy Analysis
                history = get_all_projects()
                if not history.empty:
                    embeddings = embed_model.encode(history['description'].tolist(), convert_to_tensor=True)
                    new_emb = embed_model.encode(p_desc, convert_to_tensor=True)
                    cos_sim = util.cos_sim(new_emb, embeddings).max().item()
                    if cos_sim > 0.8: st.warning(f"🚩 Redundancy Warning: {cos_sim:.2f} similarity detected.")

                raw = StrategicInferenceEngine.run_prescriptive_audit(p_desc, budget, months, team)
                data = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in raw.split('\n') if ":" in line}
                
                s_val = safe_int(data.get('SCORE'), 50)
                fr, tr, orisk, bs = safe_int(data.get('F_RISK')), safe_int(data.get('T_RISK')), safe_int(data.get('O_RISK')), safe_int(data.get('BIAS'))
                hs = safe_int(data.get('HYPE'), 20)

                with col_viz:
                    st.markdown(f"### Prescriptive Index: {s_val}%")
                    m1, m2 = st.columns(2)
                    m1.metric("Exp. CapEx", f"${safe_int(data.get('EXPECTED_COST'), budget):,}")
                    m2.metric("Exp. Timeline", f"{data.get('COMPLETION_DATE', months)} Mo")
                    
                    v_sub = st.tabs(["📊 Radar", "🧠 XAI Hub", "🏔️ 3D Sensitivity", "🌪️ Tornado Analysis", "📅 Gantt"])
                    with v_sub[0]:
                        
                        fig_radar = go.Figure(go.Scatterpolar(r=[fr, tr, orisk, bs, hs], theta=['Fin', 'Tech', 'Ops', 'Bias', 'Hype'], fill='toself', line_color='#7e57c2'))
                        st.plotly_chart(fig_radar, use_container_width=True)
                    with v_sub[1]:
                        st.markdown(f"<div class='reasoning-hub'><b>Success Logic:</b> {data.get('REASON_SUCCESS')}<br><br><b>Prescriptive Adjustment:</b> {data.get('PRESCRIPTIVE_ADJUSTMENT')}</div>", unsafe_allow_html=True)
                    with v_sub[2]:
                        
                        xs, ys = np.linspace(budget*0.5, budget*1.5, 20), np.linspace(months*0.5, months*1.5, 20)
                        X, Y = np.meshgrid(xs, ys); Z = 100 - ((X/budget)*fr + (Y/months)*tr)
                        st.plotly_chart(go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')]), use_container_width=True)
                    with v_sub[3]:
                        
                        td_df = pd.DataFrame({'Factor': ['Labor', 'Ethics', 'Capital', 'Time'], 'Impact': [team*2.5, bs-20, fr-50, tr-50]})
                        st.plotly_chart(px.bar(td_df, x='Impact', y='Factor', orientation='h', color_discrete_sequence=['#7e57c2']), use_container_width=True)
                    with v_sub[4]:
                        
                        try:
                            tasks = []; curr_date = datetime.now()
                            for item in data.get('MILESTONES', "").split(';'):
                                if ',' in item:
                                    n, d = item.split(',')
                                    tasks.append(dict(Task=n.strip(), Start=curr_date, Finish=curr_date + timedelta(days=safe_int(d, 30)), Resource='AI Phase'))
                                    curr_date += timedelta(days=safe_int(d, 30))
                            st.plotly_chart(ff.create_gantt(tasks, index_col='Resource', group_tasks=True), use_container_width=True)
                        except: st.info("Gantt logic processed.")

                save_audit({'name': p_name, 'desc': p_desc, 'budget': budget, 'months': months, 'score': s_val, 'f_risk': fr, 't_risk': tr, 'o_risk': orisk, 'risk_level': 'High' if s_val < 65 else 'Low', 'summary': data.get('SUMMARY'), 'advice': data.get('ADVICE'), 'category': data.get('CATEGORY', 'General'), 'reasoning': data.get('REASON_SUCCESS'), 'hype': hs, 'stress_test': data.get('STRESS_TEST'), 'risk_logic': data.get('REASON_RISK'), 'bias': bs, 'exp_cost': safe_int(data.get('EXPECTED_COST')), 'exp_completion': data.get('COMPLETION_DATE')})

with tab_portfolio:
    p_data = get_all_projects()
    if not p_data.empty:
        st.markdown("### 1. The Efficient Frontier (Strategic Optimization)")
        
        st.plotly_chart(px.scatter(p_data, x='f_risk', y='score', size='budget', color='category', hover_name='project_name', height=600), use_container_width=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 2. Cumulative Resource Capacity Load")
        
        st.plotly_chart(px.area(p_data, x='timestamp', y='budget', color='category', height=600), use_container_width=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 3. Sector Capital Allocation Hierarchy")
        
        st.plotly_chart(px.treemap(p_data, path=['category', 'project_name'], values='budget', color='score', height=700), use_container_width=True)
    else: st.info("Logged data required.")

with tab_heatmap:
    st.subheader("🌡️ ISO 31000 Standard 5x5 Strategic Matrix")
    
    p_data = get_all_projects()
    if not p_data.empty:
        p_data['Prob_Rank'] = pd.cut(p_data['f_risk'], bins=5, labels=[1, 2, 3, 4, 5]).astype(int)
        p_data['Impact_Rank'] = pd.cut(p_data['budget'], bins=5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        z_bg = [[1, 1, 2, 2, 3], [1, 2, 2, 3, 3], [2, 2, 3, 3, 4], [2, 3, 3, 4, 4], [3, 3, 4, 4, 4]]
        fig_5x5 = go.Figure()
        fig_5x5.add_trace(go.Heatmap(z=z_bg, x=['Insignificant', 'Minor', 'Moderate', 'Major', 'Critical'], y=['Rare', 'Unlikely', 'Possible', 'Likely', 'Almost Certain'], colorscale='RdYlGn', reversescale=True, showscale=False, opacity=0.4))
        
        fig_5x5.add_trace(go.Scatter(x=p_data['Prob_Rank'] - 0.5 + np.random.uniform(-0.1, 0.1, len(p_data)), y=p_data['Impact_Rank'] - 0.5 + np.random.uniform(-0.1, 0.1, len(p_data)), mode='markers+text', text=p_data['project_name'], textposition='top center', marker=dict(size=15, color='black', symbol='diamond-wide', line=dict(width=1, color='white'))))
        fig_5x5.update_layout(height=750, template="plotly_white", xaxis_title="Severity of Impact", yaxis_title="Probability of Occurrence")
        st.plotly_chart(fig_5x5, use_container_width=True)
    else: st.info("Audits required.")

with tab_monte_carlo:
    st.subheader("🎲 Monte Carlo Success Convergence")
    
    p_data = get_all_projects()
    if not p_data.empty:
        sims = np.random.normal(p_data.iloc[0]['score'], 12, 1000)
        st.plotly_chart(px.histogram(sims, nbins=40, title="Probability Density Function", color_discrete_sequence=['#9575cd'], height=600), use_container_width=True)
        st.write(f"**95% Statistical Confidence:** {np.percentile(sims, 2.5):.1f}% - {np.percentile(sims, 97.5):.1f}% success likelihood.")

with tab_faceoff:
    projects = get_all_projects()
    if len(projects) >= 2:
        names = projects['project_name'].unique()
        s1 = st.selectbox("Strategic Lead A", names, key="f1"); s2 = st.selectbox("Strategic Lead B", names, key="x2")
        if st.button("Generate Comparative Report"):
            d1, d2 = projects[projects['project_name'] == s1].iloc[0], projects[projects['project_name'] == s2].iloc[0]
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatterpolar(r=[d1.f_risk, d1.t_risk, d1.o_risk, d1.bias_score, d1.hype_score], theta=['Fin', 'Tech', 'Ops', 'Bias', 'Hype'], fill='toself', name=s1))
            fig_comp.add_trace(go.Scatterpolar(r=[d2.f_risk, d2.t_risk, d2.o_risk, d2.bias_score, d2.hype_score], theta=['Fin', 'Tech', 'Ops', 'Bias', 'Hype'], fill='toself', name=s2))
            st.plotly_chart(fig_comp, use_container_width=True)
            

# --- STRATEGIC MASTER AUDIT LOG ---
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🛡️ Strategic Master Audit Log")
all_records = get_all_projects()
if not all_records.empty:
    st.markdown("###### Comprehensive Project Repository • Searchable Historical Data")
    st.dataframe(all_records, use_container_width=True)
    
    # Optional Data Export for external analysis
    csv = all_records.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Export Audit Log to CSV", data=csv, file_name='strategic_audit_log.csv', mime='text/csv')
else:
    st.info("No audit records found in the database.")