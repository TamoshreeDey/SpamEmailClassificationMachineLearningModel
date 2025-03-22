from flask import Flask, request, jsonify, render_template
import joblib
import os

# Initialize Flask app
app = Flask(__name__, template_folder="template", static_folder="static")

# Define model and vectorizer paths
model_path = "model/spam_classifier.joblib"
vectorizer_path = "model/vectorizer.joblib"

# Load the trained model and vectorizer
print(f"Loading model from: {model_path}")
print(f"Loading vectorizer from: {vectorizer_path}")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("Model and vectorizer loaded successfully! 🎉")

# Home route (renders HTML page)
@app.route("/")
def home():
    return render_template("index.html")

# Classification route
@app.route("/classify", methods=["POST"])
def classify_email():
    data = request.get_json()
    email_text = data.get("email", "").strip()

    if not email_text:
        return jsonify({"error": "No email text provided"}), 400

    # Transform email text using the vectorizer
    email_tfidf = vectorizer.transform([email_text])

    # Predict using the model
    prediction = model.predict(email_tfidf)[0]
    spam_probability = model.predict_proba(email_tfidf)[0][1]  # Probability of being spam

    # Format the output
    result = {
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "spam_probability": float(spam_probability)  # Convert to regular float
    }

    return jsonify(result)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
