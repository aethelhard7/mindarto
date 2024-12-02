import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
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
    /* General Background */
    .stApp {
        background-color: #f0f0f5;  /* Soft light background for the whole app */
    }

    /* Styling for the header titles */
    h1, h2, h3 {
        color: #1d3557;
    }

    /* Sidebar background */
    .stSidebar {
        background-color: #457b9d;
        color: white;
    }

    /* Styling for the page navigation */
    .stSidebar .stSelectbox, .stSidebar .stButton {
        color: #fff;
        background-color: #1d3557;
    }

    /* Styling for buttons */
    .stButton {
        background-color: #1d3557;
        color: #fff;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton:hover {
        background-color: #457b9d;
    }

    /* Custom styling for Markdown */
    .stMarkdown {
        background-color: #a8dadc;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        color: #333;
    }

    /* Regression Score Background */
    .stMarkdown.regression-score {
        background-color: #f1faee;  /* Soft pastel for regression results */
        color: #333;
        padding: 15px;
        border-radius: 8px;
    }

    /* DataFrame Styling */
    .stDataFrame {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Styling for the page title */
    .stTitle {
        font-weight: bold;
        font-size: 32px;
        color: #1d3557;
    }

    /* Table Styling */
    .stTable {
        color: #333;
    }

    /* Styling for Success and Error Messages */
    .stSuccess, .stError {
        color: #fff;
        background-color: #2d6a4f; /* Green for success */
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 16px;
    }
    .stError {
        background-color: #d90429; /* Red for error */
    }

    /* Styling for the input labels */
    .stTextInput, .stNumberInput {
        color: #333;
        background-color: #f1faee;
        border-radius: 5px;
        border: 1px solid #d1d1d6;
    }

    .stTextInput input, .stNumberInput input {
        color: #333;
    }

    .stNumberInput label, .stTextInput label {
        color: #1d3557;
        font-weight: bold;
    }

    /* Styling for messages */
    .stMessage {
        color: #333;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Beranda", "Analisis Data Eksploratif", "Visualisasi", "Machine Learning", "Prediksi"]
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
        Selamat datang di aplikasi **Prediksi Kualitas Pisang**. Aplikasi ini memungkinkan Anda untuk mengeksplorasi dataset pisang
        dan menggunakan machine learning untuk memprediksi kualitas pisang berdasarkan berbagai fitur.
        Gunakan sidebar untuk menavigasi ke bagian lain.
        """
    )

# --- Page: Analisis Data Eksploratif ---
elif page == "Analisis Data Eksploratif":
    st.title("Analisis Data Eksploratif (EDA)")
    st.write("### Tampilan Dataset")
    st.dataframe(dataset)

    st.write("### Statistik Deskriptif")
    st.write(dataset.describe())

    st.write("### Cek Nilai yang Hilang")
    missing_values = dataset.isnull().sum()
    st.write(missing_values[missing_values > 0])

    st.write("### Data Kategorikal")
    categorical_columns = ['variety', 'region', 'quality_category', 'ripeness_category']
    for col in categorical_columns:
        st.write(f"### Distribusi {col}")
        st.write(dataset[col].value_counts())

# --- Page: Visualisasi ---
elif page == "Visualisasi":
    st.title("Visualisasi Interaktif")

    # Scatter plot dengan Plotly
    fig = px.scatter(dataset, x="quality_score", y="sugar_content_brix", color="region", title="Skor Kualitas vs Kandungan Gula (Brix)")
    st.plotly_chart(fig)

    # Histogram dengan Plotly
    st.write("### Histogram Skor Kualitas")
    fig_hist = px.histogram(dataset, x="quality_score", nbins=30, title="Distribusi Skor Kualitas")
    st.plotly_chart(fig_hist)

    # Box Plot untuk Firmness vs Ripeness Category
    st.write("### Firmness vs Kategori Kematangan (Box Plot)")
    fig_box = px.box(dataset, x="ripeness_category", y="firmness_kgf", color="ripeness_category", title="Firmness vs Kategori Kematangan")
    st.plotly_chart(fig_box)

# --- Page: Machine Learning ---
elif page == "Machine Learning":
    st.title("Latih Model Machine Learning")

    # Pilih fitur dan target
    st.write("### Pilih Fitur dan Target")
    features = st.multiselect("Pilih Fitur", dataset.columns, default=['quality_score', 'ripeness_index', 'sugar_content_brix'])
    target = st.selectbox("Pilih Target", dataset.columns)

    if not features:
        st.warning("Silakan pilih minimal satu fitur.")
    elif target in features:
        st.error("Target tidak boleh dipilih sebagai salah satu fitur.")
    else:
        st.write(f"Melatih model untuk memprediksi {target} menggunakan fitur {features}.")

        # Pisahkan dataset menjadi data pelatihan dan pengujian
        X = dataset[features]
        y = dataset[target]

        # Normalisasi fitur
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Tentukan jenis model berdasarkan tipe target
        if dataset[target].dtype == 'object' or len(dataset[target].unique()) < 10:
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Akurasi Model Klasifikasi: **{acc:.2f}**")
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2_score = model.score(X_test, y_test)
            st.markdown(f"### Skor R² Model Regresi: **{r2_score:.2f}**", unsafe_allow_html=True)

        # Simpan model dan scaler di session_state
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.features = features
        st.session_state.target = target
        st.success("Model dan scaler berhasil disimpan untuk digunakan di halaman prediksi.")

# --- Page: Prediksi ---
elif page == "Prediksi":
    st.title("Prediksi Kualitas Pisang")

    st.write("### Masukkan Data untuk Prediksi")

    # Periksa apakah model dan scaler telah disimpan
    if "model" not in st.session_state or "scaler" not in st.session_state:
        st.error("Model dan Scaler belum didefinisikan. Silakan latih model terlebih dahulu di halaman Machine Learning.")
    else:
        # Ambil fitur yang telah dilatih
        features = st.session_state.features
        target = st.session_state.target
        scaler = st.session_state.scaler
        model = st.session_state.model

        # Kumpulkan input data dari pengguna
        input_data = {}
        for feature in features:
            input_data[feature] = st.number_input(f"Masukkan nilai untuk {feature}", value=0.0, step=0.1)

        if st.button("Prediksi Kualitas"):
            # Persiapkan data untuk prediksi
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            # Prediksi
            prediction = model.predict(input_scaled)

            # Tampilkan hasil prediksi
            if dataset[target].dtype == 'object' or len(dataset[target].unique()) < 10:
                st.write(f"Hasil Prediksi untuk **{target}**: **{prediction[0]}**")
            else:
                st.write(f"Hasil Prediksi untuk **{target}**: **{prediction[0]:.2f}**")
