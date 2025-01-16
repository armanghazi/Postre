# Postre
# Menu Dataset Classification App

This Streamlit application predicts your favorite dessert based on your selected menu preferences (starter, main course, drink, etc.). It uses a Ridge Classifier model for the prediction and displays the corresponding dessert image.

## Features
- **User Input**: Choose your preferences for:
  - Gender
  - Age
  - Starter
  - Main Course
  - Drink
- **Prediction**: The app predicts your favorite dessert based on the input menu preferences.
- **Visualization**: Displays an image of the predicted dessert.

## How It Works
1. **Dataset**: The app uses a menu dataset (`menu_dataset (1).csv`) with food preferences to train the Ridge Classifier model.
2. **Model Training**: The dataset is preprocessed, and a Ridge Classifier is trained to predict dessert categories.
3. **User Interaction**: Users provide input via dropdown menus and sliders.
4. **Dessert Prediction**: The app predicts the dessert and fetches the corresponding image from a GitHub-hosted image repository.

## Technologies Used
- **Streamlit**: For building the interactive web app.
- **Pandas**: For data preprocessing.
- **Scikit-learn**: For training the Ridge Classifier model.
- **Pillow (PIL)**: For handling and displaying images.

## Project Structure
├── app.py # Main Streamlit app script ├── menu_dataset (1).csv # Dataset used for training ├── img/ # Folder containing dessert images │ ├── flan.jpg │ ├── helado.jpg │ ├── fruta-fresca.jpg │ ├── tarta-queso.jpg │ ├── brownie.jpg │ ├── queso-membrillo.jpg

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/ghaziaskari/Postre.git
   cd Postre
install dependencies:
pip install -r requirements.txt

Run the app locally:

streamlit run model_ridge_image3.py

    Open the app in your browser at http://localhost:8501.

Visit the Deployed App

You can try the app online here:
Menu Dataset Classification App(https://postre2.streamlit.app/)
Example Output

    Input: Gender: Masculino, Age: 30, Starter: Croquetas, Main Course: Lentejas, Drink: Cerveza.
    Prediction: Flan.
    Visualization: Displays an image of Flan.

Contributing

Feel free to submit issues or pull requests to improve the app.
License

This project is licensed under the MIT License.


### Notes:
1. Update the placeholder GitHub repository link (`https://github.com/ghaziaskari/Postre.git`) if needed.
2. Replace the license section with your preferred license, if different.

Let me know if you'd like further adjustments! 🚀

