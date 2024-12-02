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

# --- Page: Beranda ---
if page == "Beranda":
    st.title("🍌 Prediksi Kualitas Pisang")
    st.write(
        """
        Selamat datang di aplikasi **Prediksi Kualitas Pisang**. 
        Aplikasi ini dirancang untuk membantu Anda menganalisis dan memprediksi kualitas pisang 
        berdasarkan berbagai fitur seperti kandungan gula, kematangan, dan lainnya.
        
        Fitur utama aplikasi ini:
        - Eksplorasi data melalui analisis deskriptif
        - Visualisasi interaktif untuk mempermudah pemahaman
        - Model pembelajaran mesin untuk prediksi kualitas
        - Kemudahan prediksi dengan masukan data manual
        
        Gunakan menu di samping untuk memulai.
        """
    )

# --- Page: Analisis Data ---
elif page == "Analisis Data":
    st.title("Analisis Data Eksploratif (EDA)")
    st.write("### Tampilan Dataset")
    st.dataframe(dataset, height=400, use_container_width=True)

    st.write("### Statistik Deskriptif dan Nilai Hilang")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Statistik Deskriptif")
        st.write(dataset.describe())
    with col2:
        st.write("#### Pemeriksaan Nilai Hilang")
        missing_values = dataset.isnull().sum()
        st.write(missing_values[missing_values > 0])

    st.write("### Sebaran Data Kategorikal")
    categorical_columns = ['variety', 'region', 'quality_category', 'ripeness_category']
    for col in categorical_columns:
        st.write(f"#### Sebaran {col}")
        st.write(dataset[col].value_counts())

# --- Page: Visualisasi Data ---
elif page == "Visualisasi Data":
    st.title("Visualisasi Interaktif")

    # Scatter plot dengan Plotly
    st.write("### Scatter Plot")
    fig = px.scatter(
        dataset, x="quality_score", y="sugar_content_brix", color="region",
        title="Skor Kualitas vs Kandungan Gula (Brix)",
        template="plotly_white"
    )
    st.plotly_chart(fig)

    # Histogram
    st.write("### Histogram Skor Kualitas")
    fig_hist = px.histogram(dataset, x="quality_score", nbins=30, title="Distribusi Skor Kualitas", template="plotly_white")
    st.plotly_chart(fig_hist)

    # Box Plot
    st.write("### Kekerasan (Firmness) Berdasarkan Kategori Kematangan")
    fig_box = px.box(
        dataset, x="ripeness_category", y="firmness_kgf", color="ripeness_category",
        title="Firmness Berdasarkan Kategori Kematangan", template="plotly_white"
    )
    st.plotly_chart(fig_box)

# --- Page: Latih Model ---
elif page == "Latih Model":
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
                st.success(f"Skor R² Model Regresi: **{r2_score:.2f}**")

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.features = features
            st.session_state.target = target

# --- Page: Prediksi Kualitas ---
elif page == "Prediksi Kualitas":
    st.title("Prediksi Kualitas Pisang")

    if "model" not in st.session_state or "scaler" not in st.session_state:
        st.error("Model belum dilatih. Silakan latih model terlebih dahulu di halaman 'Latih Model'.")
    else:
        features = st.session_state.features
        scaler = st.session_state.scaler
        model = st.session_state.model

        st.write("### Masukkan Data untuk Prediksi")
        input_data = {feature: st.number_input(f"Masukkan nilai untuk {feature}", value=0.0, step=0.1) for feature in features}

        if st.button("Prediksi Kualitas"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)

            if dataset[st.session_state.target].dtype == 'object' or len(dataset[st.session_state.target].unique()) < 10:
                st.success(f"Hasil Prediksi: **{prediction[0]}**")
            else:
                st.success(f"Hasil Prediksi: **{prediction[0]:.2f}**")
