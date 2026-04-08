import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide", page_title="Bank Churn Dashboard")

# ---------------- LOAD FILES ----------------
model = pickle.load(open('churn_model.pkl', 'rb'))
df = pd.read_csv('European_Bank.csv')

# ---------------- GLOBAL STYLING ----------------
st.markdown("""
<style>
.main { background-color: #f4f7fb; }

.header-container {
    background: linear-gradient(90deg, #ffffff, #eef2f7);
    padding: 18px 30px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

h1 {
    text-align: center;
    font-size: 30px;
    font-weight: 600;
    color: #1f2937;
}

.center-text {
    text-align: center;
    font-size: 14px;
    color: #6b7280;
}

.section-header {
    font-size: 22px;
    font-weight: 600;
    margin-top: 35px;
    margin-bottom: 10px;
    color: #1f2937;
}

.sub-header {
    font-size: 16px;
    font-weight: 500;
    margin-top: 10px;
    color: #374151;
}

.divider {
    height: 2px;
    background: linear-gradient(to right, #2563eb, #60a5fa);
    border-radius: 10px;
    margin: 25px 0;
}

img {
    display: block;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="header-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.3, 3.4, 1.3])

with col1:
    st.image("logo.png", width=130)

with col2:
    st.markdown("""
    <h1>Predictive Modeling and Risk Scoring for Bank Customer Churn</h1>
    <div class="center-text">
    Prepared by <b>Geetanjali Sateesh</b> | Mentored by <b>Saiprasad Kagne</b>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.image("ecb_logo.png", width=130)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Customer Inputs")

age = st.sidebar.slider("Age", 18, 100, 30)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, 600)
balance = st.sidebar.number_input("Balance", 0.0, 300000.0, 50000.0)
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)

st.sidebar.subheader("Engagement & Products")
num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
is_active = st.sidebar.selectbox("Active Member", [0,1])
has_card = st.sidebar.selectbox("Credit Card", [0,1])
gender = st.sidebar.selectbox("Gender (Male=1)", [1,0])
geo = st.sidebar.selectbox("Geography", ["France","Germany","Spain"])

# ---------------- VIEW MODE ----------------
view_mode = st.sidebar.radio("View Mode", ["Full Dashboard", "Use Navigation"])

if view_mode == "Use Navigation":
    page = st.sidebar.radio("Go to", [
        "Overview",
        "Module 1",
        "Module 2",
        "Module 3",
        "Module 4",
        "Module 5"
    ])
else:
    page = "Full"

# ---------------- FEATURE ENGINEERING ----------------
geo_germany = 1 if geo=="Germany" else 0
geo_spain = 1 if geo=="Spain" else 0

balance_salary_ratio = balance/(salary+1)
product_density = num_products/(tenure+1)
engagement_product = is_active*num_products
age_tenure = age*tenure

input_data = np.array([[ 
    0, credit_score, age, tenure, balance,
    num_products, has_card, is_active, salary,
    geo_germany, geo_spain, gender,
    balance_salary_ratio, product_density,
    engagement_product, age_tenure
]])

prob = model.predict_proba(input_data)[0][1]

if prob < 0.3:
    risk = "Low"
    color = "green"
elif prob < 0.6:
    risk = "Medium"
    color = "orange"
else:
    risk = "High"
    color = "red"

# ---------------- DATA PREP ----------------
df_model = df.copy()

df_model['Gender_Male'] = (df_model['Gender'] == 'Male').astype(int)
df_model['Geography_Germany'] = (df_model['Geography'] == 'Germany').astype(int)
df_model['Geography_Spain'] = (df_model['Geography'] == 'Spain').astype(int)

df_model = df_model.drop(['Gender','Geography','CustomerId','Surname'], axis=1)

df_model['BalanceSalaryRatio'] = df_model['Balance']/(df_model['EstimatedSalary']+1)
df_model['ProductDensity'] = df_model['NumOfProducts']/(df_model['Tenure']+1)
df_model['EngagementProductInteraction'] = df_model['IsActiveMember']*df_model['NumOfProducts']
df_model['AgeTenureInteraction'] = df_model['Age']*df_model['Tenure']

expected_columns = [
    'Year','CreditScore','Age','Tenure','Balance','NumOfProducts',
    'HasCrCard','IsActiveMember','EstimatedSalary',
    'Geography_Germany','Geography_Spain','Gender_Male',
    'BalanceSalaryRatio','ProductDensity',
    'EngagementProductInteraction','AgeTenureInteraction'
]

for col in expected_columns:
    if col not in df_model.columns:
        df_model[col] = 0

X_full = df_model[expected_columns]

df['Churn_Probability'] = model.predict_proba(X_full)[:,1]
df['Risk'] = pd.cut(df['Churn_Probability'], bins=[0,0.3,0.6,1], labels=['Low','Medium','High'])

# ================= OVERVIEW =================
if page == "Overview" or page == "Full":
    st.markdown('<div class="section-header">📊 Overview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Average Risk", f"{df['Churn_Probability'].mean():.2f}")
    col3.metric("High Risk %", f"{(df['Risk']=='High').mean()*100:.1f}%")

    st.markdown("**Overall Churn Risk Distribution**")
    st.plotly_chart(px.histogram(df, x='Churn_Probability', title="Distribution of Churn Probability"), use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ================= MODULE 1 =================
if page == "Module 1" or page == "Full":
    st.markdown('<div class="section-header">🧩 Module 1: Customer Churn Risk Calculator</div>', unsafe_allow_html=True)

    st.metric("Churn Probability", f"{prob:.2f}")
    st.markdown(f"**Risk Level:** <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)

    st.plotly_chart(go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Customer Churn Risk Score"},
        gauge={'axis': {'range': [0,1]}, 'bar': {'color': color}}
    )), use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ================= MODULE 2 =================
if page == "Module 2" or page == "Full":
    st.markdown('<div class="section-header">📊 Module 2: Probability Distribution Analysis</div>', unsafe_allow_html=True)

    st.markdown("**Histogram of Churn Probability**")
    st.plotly_chart(px.histogram(df, x='Churn_Probability', nbins=30,
                                title="Churn Probability Histogram"), use_container_width=True)

    st.markdown("**Density & Spread of Churn Probability**")
    st.plotly_chart(px.histogram(df, x='Churn_Probability', marginal='box',
                                title="Density and Box Plot of Churn Probability"), use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ================= MODULE 3 =================
if page == "Module 3" or page == "Full":
    st.markdown('<div class="section-header">🌍 Module 3: Customer Segmentation Insights</div>', unsafe_allow_html=True)

    st.markdown("**Customer Risk Segmentation**")
    st.plotly_chart(px.pie(df, names='Risk', title="Customer Risk Categories"), use_container_width=True)

    st.markdown("**Churn Rate by Geography**")
    geo_churn = df.groupby('Geography')['Exited'].mean().reset_index()
    st.plotly_chart(px.bar(geo_churn, x='Geography', y='Exited',
                           title="Geographical Churn Analysis"), use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ================= MODULE 4 =================
if page == "Module 4" or page == "Full":
    st.markdown('<div class="section-header">📈 Module 4: Feature Importance Dashboard</div>', unsafe_allow_html=True)

    st.markdown("**Top Factors Influencing Customer Churn**")
    st.plotly_chart(px.bar(x=[0.30,0.25,0.10,0.08,0.07],
                           y=["Products","Age","Activity","Balance","Geography"],
                           orientation='h',
                           title="Feature Importance Ranking"), use_container_width=True)

    st.markdown("**SHAP Summary Plot for Model Explainability**")
    st.image("shap_summary.png")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ================= MODULE 5 =================
if page == "Module 5" or page == "Full":
    st.markdown('<div class="section-header">🎮 Module 5: What-if Scenario Simulator</div>', unsafe_allow_html=True)

    st.markdown("**Comparison of Baseline vs Current Customer Risk**")

    fig_compare = go.Figure(data=[
        go.Bar(name='Baseline Risk', x=['Customer'], y=[0.5]),
        go.Bar(name='Current Risk', x=['Customer'], y=[prob])
    ])

    fig_compare.update_layout(barmode='group',
                             title="Impact of Input Changes on Churn Probability")

    st.plotly_chart(fig_compare, use_container_width=True)