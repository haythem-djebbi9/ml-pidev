import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

# Base de données de formation (symptômes + médicaments internationaux)
symptoms = [
    "i have a headache", "i am coughing a lot", "i have a fever", 
    "i have abdominal pain", "i feel nauseous", "i have a urinary tract infection",
    "i have muscle pain", "i have digestive issues", "i have a skin allergy",
    "i have a sore throat", "i am feeling dizzy", "i have diarrhea",
    "i have a cold", "i am sneezing a lot", "i have chest pain"
]

medications = [
    "Acetaminophen", "Dextromethorphan", "Ibuprofen", 
    "Hyoscine butylbromide", "Ondansetron", "Amoxicillin",
    "Ibuprofen", "Probiotics", "Hydrocortisone cream",
    "Amoxicillin", "Meclizine", "Loperamide",
    "Pseudoephedrine", "Cetirizine", "Doliprane"
]

# Fonction de prétraitement pour normaliser les entrées
def preprocess_text(text):
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    return text

# Appliquer le prétraitement aux symptômes
symptoms = [preprocess_text(symptom) for symptom in symptoms]

# Transformer les textes en vecteurs
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(symptoms)

# Entraîner un modèle Naïve Bayes
model = MultinomialNB()
model.fit(X, medications)

# Sauvegarder le modèle et le vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Modèle entraîné et sauvegardé avec succès !")

# Test rapide du modèle (optionnel)
if __name__ == "__main__":
    test_input = "i have a fever"
    test_input_processed = preprocess_text(test_input)
    X_test = vectorizer.transform([test_input_processed])
    prediction = model.predict(X_test)[0]
    print(f"Symptôme : {test_input}")
    print(f"Médicament recommandé : {prediction}")