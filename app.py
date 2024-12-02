import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Set up the Streamlit page
st.set_page_config(
    page_title="Prediksi Kualitas Pisang",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    /* Set background and font colors for better contrast */
    .stApp {
        background-color: #f0f0f5;
        color: #333;
    }

    .stTitle {
        color: #1d3557;
        font-weight: bold;
    }

    /* Sidebar background */
    .stSidebar {
        background-color: #1d3557;
        color: white;
    }

    /* Sidebar elements (buttons, inputs) */
    .stSidebar .stButton, .stSidebar .stSelectbox, .stSidebar .stMultiselect {
        background-color: #457b9d;
        color: white;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #1d3557;
    }

    /* Streamlit widget styling */
    .stButton, .stSlider, .stNumberInput, .stSelectbox {
        background-color: #457b9d;
        color: white;
    }

    /* Add hover effects */
    .stButton:hover {
        background-color: #1d3557;
        color: white;
    }

    /* Plotly chart background */
    .plotly-graph-div {
        background-color: #ffffff;
    }

    /* Navigation page background */
    .stSidebar selectbox {
        background-color: #f1faee;
        color: #333;
    }

    /* Background for the model accuracy score */
    .stMarkdown {
        background-color: #a8dadc;
        padding: 10px;
        border-radius: 5px;
        color: #333;  /* Make sure the text is visible */
        font-weight: bold;
    }

    /* Styling for regression scores specifically */
    .stMarkdown.regression-score {
        background-color: #ffcccb;  /* Soft red background for negative score */
        padding: 12px;
        color: #d32f2f;  /* Dark red for text */
        font-weight: bold;
        border-radius: 8px;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #457b9d;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigasi Halaman",
    ["Beranda", "Analisis Data", "Visualisasi Data", "Latih Model", "Prediksi Kualitas"]
)

# Load the dataset
@st.cache_data  # Cache the dataset loading function
def load_data():
    return pd.read_csv("banana_quality_dataset.csv")

# Load dataset
dataset = load_data()

# --- Page: Latih Model ---
if page == "Latih Model":
    st.title("Pelatihan Model Pembelajaran Mesin")

    st.write("### Pilih Fitur dan Target")
    features = st.multiselect("Pilih Fitur", dataset.columns, default=['quality_score', 'ripeness_index', 'sugar_content_brix'])
    target = st.selectbox("Pilih Target", dataset.columns)

    if not features:
        st.warning("Pilih setidaknya satu fitur.")
    elif target in features:
        st.error("Target tidak boleh dipilih sebagai salah satu fitur.")
    else:
        st.write(f"Melatih model untuk memprediksi {target} menggunakan fitur {features}.")
        X = dataset[features]
        y = dataset[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        with st.spinner("Model sedang dilatih..."):
            if dataset[target].dtype == 'object' or len(dataset[target].unique()) < 10:
                model = RandomForestClassifier(n_estimators=100)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Akurasi Model Klasifikasi: **{acc:.2f}**")
            else:
                model = RandomForestRegressor(n_estimators=100)
                model.fit(X_train, y_train)
                r2_score = model.score(X_test, y_test)
                
                # Apply the custom style for the regression score
                if r2_score < 0:
                    st.markdown(f"### Skor R² Model Regresi: **{r2_score:.2f}**", unsafe_allow_html=True)
                    st.markdown(f'<div class="stMarkdown regression-score">Skor R² Model Regresi: **{r2_score:.2f}**</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f"### Skor R² Model Regresi: **{r2_score:.2f}**", unsafe_allow_html=True)

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.features = features
            st.session_state.target = target
