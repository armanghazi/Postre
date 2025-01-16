import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
import os

# Define the path for images
IMG_PATH = "img"

def load_data(filepath):
    """Load and preprocess the menu dataset."""
    data = pd.read_csv(filepath, sep=";")

    # Mappings
    primeros_cat = {
        "Risotto": "arroz",
        "Paella": "arroz",
        "Arroz 3 delicias": "arroz",
        "Lasaña": "pasta",
        "Macarrones": "pasta",
        "Crema de verduras": "verduras",
        "Lentejas": "legumbres",
        "Garbanzos con espinacas": "legumbres",
        "Sopa de pollo": "sopa"
    }
    segundos_cat = {
        "Filete de ternera": "carne",
        "Merluza al horno": "pescado",
        "Pollo asado": "carne",
        "Hamburguesa": "carne",
        "Costillas BBQ": "carne",
        "Albóndigas": "carne",
        "Cachopo": "carne",
        "Chuletón": "carne",
        "Pollo al curry": "carne",
        "Salmón": "pescado",
        "Bacalao": "pescado"
    }
    postre_cat = {
        "Flan": 0,
        "Helado": 1,
        "Fruta fresca": 2,
        "Tarta de queso": 3,
        "Brownie": 4,
        "Queso y membrillo": 5
    }

    # Encode categorical columns
    data['Hombre'] = (data['Género'] == "Masculino").astype(int)
    data['Mujer'] = (data['Género'] == "Femenino").astype(int)

    # Encode entrantes and bebidas
    data = pd.concat([data, pd.get_dummies(data['Entrante'], prefix='Entrante')], axis=1)
    data = pd.concat([data, pd.get_dummies(data['Bebida'], prefix='Bebida')], axis=1)

    # Encode primeros and segundos
    data['categoria_primero'] = data['Primer Plato'].map(primeros_cat)
    data['categoria_segundo'] = data['Segundo Plato'].map(segundos_cat)
    data['arroz'] = (data['categoria_primero'] == 'arroz').astype(int)
    data['pasta'] = (data['categoria_primero'] == 'pasta').astype(int)
    data['verduras'] = (data['categoria_primero'] == 'verduras').astype(int)
    data['legumbres'] = (data['categoria_primero'] == 'legumbres').astype(int)
    data['sopa'] = (data['categoria_primero'] == 'sopa').astype(int)
    data['carne'] = (data['categoria_segundo'] == 'carne').astype(int)
    data['pescado'] = (data['categoria_segundo'] == 'pescado').astype(int)

    # Encode target column
    data['target_label'] = data['Postre'].map(postre_cat)

    # Drop unused columns
    data.drop(columns=['Entrante', 'Primer Plato', 'Segundo Plato', 'Bebida', 'Género',
                       'categoria_primero', 'categoria_segundo', 'Postre'], inplace=True)

    return data

def train_model(X_train, y_train):
    """Train the Ridge Classifier model."""
    model = RidgeClassifier(alpha=14.38449888287663)
    model.fit(X_train, y_train)
    return model

def predict_dessert(model, user_inputs, columns):
    """Predict the dessert based on user inputs."""
    user_df = pd.DataFrame([user_inputs], columns=columns)
    prediction = model.predict(user_df)[0]
    reverse_postre_cat = {
        0: "Flan",
        1: "Helado",
        2: "Fruta fresca",
        3: "Tarta de queso",
        4: "Brownie",
        5: "Queso y membrillo"
    }
    return reverse_postre_cat[prediction]

def display_image(dessert_name):
    """Display an image for a dessert using the raw URL."""
    dessert_image_map = {
        "Flan": "https://raw.githubusercontent.com/ghaziaskari/Postre/main/img/flan.jpg",
        "Helado": "https://raw.githubusercontent.com/ghaziaskari/Postre/main/img/helado.jpg",
        "Fruta fresca": "https://raw.githubusercontent.com/ghaziaskari/Postre/main/img/fruta-fresca.jpg",
        "Tarta de queso": "https://raw.githubusercontent.com/ghaziaskari/Postre/main/img/tarta-queso.jpg",
        "Brownie": "https://raw.githubusercontent.com/ghaziaskari/Postre/main/img/brownie.jpg",
        "Queso y membrillo": "https://raw.githubusercontent.com/ghaziaskari/Postre/main/img/queso-membrillo.jpg"
    }

    # Get the image URL for the dessert
    url = dessert_image_map.get(dessert_name)

    if url:
        # If the URL exists, display the image
        st.image(url, caption=dessert_name, use_container_width=True)
    else:
        # If the dessert is not in the map, show a message
        st.write(f"Image not found for: {dessert_name}")

def main():
    """Main function to build the Streamlit app."""
    st.set_page_config(page_title="Menu Dataset Classification", layout="centered")

    st.title("Aplicación de clasificación de conjuntos de datos del menú")
    st.write("Esta aplicación entrena un clasificador Ridge para predecir categorías de postres y mostrar imágenes relacionadas.")

    # Load dataset
    file_path = "menu_dataset (1).csv"

    if file_path and st.button("Cargar y entrenar"):
        data = load_data(file_path)
        X = data.drop('target_label', axis=1)
        y = data['target_label']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = train_model(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Display results
        st.subheader("Métricas de evaluación de modelos")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**F1-Score:** {f1:.2f}")
        st.write(f"**Recall:** {recall:.2f}")

        # Save the model and features for user prediction
        st.session_state['model'] = model
        st.session_state['features'] = X.columns.tolist()

    # User Input for Prediction
    if 'model' in st.session_state and 'features' in st.session_state:
        st.subheader("Predice tu postre")

        genero = st.selectbox("Elige tu Género", ["Femenino", "Masculino"], key="genero")
        edad = st.slider("Elige tu Edad", min_value=18, max_value=99, value=25, step=1, key="edad")
        entrante = st.selectbox("Elige tu Entrante", ["Ninguno","Croquetas", "Espárragos", "Ensalada César", "Jamón", "Ensaladilla", "Gazpacho", "Langostinos",  "Empanada", "Mejillones"], key="entrante")
        primer_plato = st.selectbox("Elige tu Primer Plato", ["Macarrones", "Lentejas", "Sopa de pollo", "Paella", "Menestra de verduras", "Risotto", "Arroz 3 delicias", "Lasaña", "Garbanzos con espinacas", "Crema de verduras"], key="primer_plato")
        segundo_plato = st.selectbox("Elige tu Segundo Plato", ["Pollo al curry", "Chuletón", "Albóndigas", "Bacalao", "Cachopo", "Filete de ternera", "Merluza al horno", "Salmón", "Pollo asado", "Costillas BBQ", "Hamburguesa"], key="segundo_plato")
        bebida = st.selectbox("Elige tu Bebida", ["Vino blanco", "Cerveza", "Vino tinto", "Refresco", "Agua"], key="bebida")

        # Encode user inputs
        user_inputs = {
            "Edad": edad,
            "Hombre": 1 if genero == "Masculino" else 0,
            "Mujer": 1 if genero == "Femenino" else 0
        }

        # Add one-hot encoding for entrante, bebida
        for col in st.session_state['features']:
            if col.startswith("Entrante_"):
                user_inputs[col] = 1 if f"Entrante_{entrante}" == col else 0
            elif col.startswith("Bebida_"):
                user_inputs[col] = 1 if f"Bebida_{bebida}" == col else 0

        # Add binary encoding for primeros and segundos
        primeros_cat = {"arroz": ["Risotto", "Paella", "Arroz 3 delicias"],
                        "pasta": ["Lasaña", "Macarrones"],
                        "verduras": ["Crema de verduras"],
                        "legumbres": ["Lentejas", "Garbanzos con espinacas"],
                        "sopa": ["Sopa de pollo"]}

        segundos_cat = {"carne": ["Filete de ternera", "Pollo asado", "Hamburguesa", "Costillas BBQ", "Albóndigas", "Cachopo", "Chuletón", "Pollo al curry"],
                        "pescado": ["Merluza al horno", "Salmón", "Bacalao"]}

        user_inputs.update({
            "arroz": 1 if primer_plato in primeros_cat["arroz"] else 0,
            "pasta": 1 if primer_plato in primeros_cat["pasta"] else 0,
            "verduras": 1 if primer_plato in primeros_cat["verduras"] else 0,
            "legumbres": 1 if primer_plato in primeros_cat["legumbres"] else 0,
            "sopa": 1 if primer_plato in primeros_cat["sopa"] else 0,
            "carne": 1 if segundo_plato in segundos_cat["carne"] else 0,
            "pescado": 1 if segundo_plato in segundos_cat["pescado"] else 0
        })

        # Predict and display
        predicted_dessert = predict_dessert(st.session_state['model'], user_inputs, st.session_state['features'])
        st.write(f"Tu postre favorito es: **{predicted_dessert}**")

        # Display relevant image
        dessert_image = predicted_dessert.lower().replace(" ", "-") + ".jpg"
        display_image(dessert_image)

if __name__ == "__main__":
    main()
