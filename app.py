from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

# Charger le modèle et le vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    print(f"Erreur lors du chargement du modèle ou du vectorizer : {e}")
    exit(1)

# Fonction de prétraitement pour normaliser les entrées
def preprocess_text(text):
    text = text.lower()  # Convertir en minuscules
    # text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    return text

# Fonction de prédiction avec gestion des erreurs
def predict_medication(user_input):
    user_input = preprocess_text(user_input)
    X_input = vectorizer.transform([user_input])
    predicted_probabilities = model.predict_proba(X_input)

    if max(predicted_probabilities[0]) < 0.3:
        return "I cannot identify a precise treatment. Please consult a doctor."
    
    return model.predict(X_input)[0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        if not user_input:
            return jsonify({'error': 'No symptom provided'}), 400
        
        medication = predict_medication(user_input)
        return jsonify({'medication': medication})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)