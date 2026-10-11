import os
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Base directory for loading model files and static assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.pkl')

# Initialize stop words for preprocessing
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

# Global references for model, vectorizer, and metadata
model = None
vectorizer = None
metadata = None
emotion_map = {}


def load_artifacts():
    """Load model, vectorizer, and metadata artifacts from disk."""
    global model, vectorizer, metadata, emotion_map
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(METADATA_PATH)):
        print("Warning: Model files not found! Please run 'python main.py' first to train and generate model artifacts.")
        return False

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        emotion_map = metadata.get('emotion_map', {})
        print("Model, vectorizer, and metadata loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return False


# Attempt artifact loading on startup
load_artifacts()


def preprocess_text(text: str) -> str:
    """Replicate exact text preprocessing pipeline used in main.py:
    1. Lowercase
    2. Remove punctuation
    3. Remove numbers/digits
    4. Remove non-ASCII characters (emojis)
    5. Remove stopwords using NLTK word_tokenize
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Remove punctuation
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))

    # 3. Remove digits
    new_chars = []
    for char in cleaned:
        if not char.isdigit():
            new_chars.append(char)
    cleaned = "".join(new_chars)

    # 4. Remove emojis / non-ascii characters
    ascii_chars = []
    for char in cleaned:
        if char.isascii():
            ascii_chars.append(char)
    cleaned = "".join(ascii_chars)

    # 5. Tokenize and remove stopwords
    words = word_tokenize(cleaned)
    filtered_words = [w for w in words if w not in stop_words]

    return ' '.join(filtered_words)


@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response


@app.route('/', methods=['GET'])
def index():
    """Serve static/index.html."""
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/static/<path:filename>', methods=['GET'])
def serve_static(filename):
    """Serve files from the static directory."""
    return send_from_directory(STATIC_DIR, filename)


@app.route('/predict', methods=['POST'])
def predict():
    """Accept JSON {"text": "..."}, preprocess, predict emotion with probabilities, and return results."""
    global model, vectorizer, metadata, emotion_map
    if model is None or vectorizer is None:
        if not load_artifacts():
            return jsonify({
                "error": "Model not loaded. Please run 'python main.py' to generate model artifacts."
            }), 503

    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in JSON request body."}), 400

    raw_text = data['text']
    cleaned_text = preprocess_text(raw_text)

    # Transform input text using loaded TF-IDF vectorizer
    features = vectorizer.transform([cleaned_text])

    # Predict probabilities across classes
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    # Map class indices to emotion names and construct score dictionary
    all_scores = {}
    for cls_idx, prob in zip(classes, probabilities):
        emo_name = emotion_map.get(cls_idx, str(cls_idx))
        all_scores[emo_name] = round(float(prob), 4)

    # Find highest probability emotion
    best_idx = int(probabilities.argmax())
    best_class = classes[best_idx]
    predicted_emotion = emotion_map.get(best_class, str(best_class))
    confidence = round(float(probabilities[best_idx]), 4)

    return jsonify({
        "emotion": predicted_emotion,
        "confidence": confidence,
        "all_scores": all_scores,
        "cleaned_text": cleaned_text
    })


@app.route('/stats', methods=['GET'])
def stats():
    """Return model performance metrics and dataset stats from metadata."""
    global metadata
    if metadata is None:
        if not load_artifacts():
            return jsonify({
                "error": "Metadata not loaded. Please run 'python main.py' to generate metadata."
            }), 503

    emotion_counts = metadata.get('emotion_counts', {})
    response_data = {
        "bow_accuracy": metadata.get('bow_accuracy'),
        "tfidf_accuracy": metadata.get('tfidf_accuracy'),
        "total_samples": metadata.get('total_samples'),
        "emotion_counts": emotion_counts,
        "num_emotions": len(emotion_counts) if emotion_counts else len(metadata.get('emotion_map', {}))
    }
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
