from flask import Flask, request, render_template, jsonify
import joblib
import os
import re
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import numpy as np
from sklearn.preprocessing import StandardScaler
# Download required NLTK data (only if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
app = Flask(__name__)

# Define the SpamDetectionPipeline class (same as in your training script)
class SpamDetectionPipeline:
    def __init__(self, preprocessor, feature_extractor, pca_w2v, model, balancer=None):
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.pca_w2v = pca_w2v
        self.model = model
        self.balancer = balancer
        
    def preprocess_text(self, text):
        """Preprocess single text"""
        return self.preprocessor.preprocess(text)
    
    def extract_features(self, texts):
        """Extract features from texts"""
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Extract TF-IDF features
        tfidf_features = self.feature_extractor.tfidf_vectorizer.transform(processed_texts)
        
        # Extract Word2Vec features
        w2v_features = self.feature_extractor.get_word2vec_features(processed_texts)
        w2v_reduced = self.pca_w2v.transform(w2v_features)
        
        # Combine features
        combined_features = hstack([tfidf_features, csr_matrix(w2v_reduced)])
        
        return combined_features
    
    def predict(self, texts):
        """Predict spam/ham for texts"""
        features = self.extract_features(texts)
        
        if self.model.__class__.__name__ == 'GaussianNB':
            features = features.toarray()
        
        predictions = self.model.predict(features)
        return ['spam' if pred == 1 else 'ham' for pred in predictions]
    
    def predict_proba(self, texts):
        """Predict probabilities for texts"""
        features = self.extract_features(texts)
        
        if self.model.__class__.__name__ == 'GaussianNB':
            features = features.toarray()
        
        probabilities = self.model.predict_proba(features)
        return probabilities
class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.count_vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.word2vec_model = None
        self.scaler = StandardScaler()
    def fit_count_vectorizer(self, texts):
        """Fit count vectorizer"""
        return self.count_vectorizer.fit_transform(texts)
    def fit_tfidf_vectorizer(self, texts):
        """Fit TF-IDF vectorizer"""
        return self.tfidf_vectorizer.fit_transform(texts)
    def fit_word2vec(self, texts, vector_size=100, window=5, min_count=1, workers=4):
        """Fit Word2Vec model"""
        sentences = [text.split() for text in texts]
        self.word2vec_model = Word2Vec(sentences, vector_size=vector_size, window=window, 
                                      min_count=min_count, workers=workers, seed=42)
        return self.get_word2vec_features(texts)
    def get_word2vec_features(self, texts):
        """Extract Word2Vec features"""
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not fitted yet")
        
        features = []
        for text in texts:
            words = text.split()
            word_vectors = []
            for word in words:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            
            if word_vectors:
                features.append(np.mean(word_vectors, axis=0))
            else:
                features.append(np.zeros(self.word2vec_model.vector_size))
        
        return np.array(features)
    
    def extract_all_features(self, texts):
        """Extract all types of features"""
        count_features = self.fit_count_vectorizer(texts)
        tfidf_features = self.fit_tfidf_vectorizer(texts)
        w2v_features = self.fit_word2vec(texts)
        return {
            'count': count_features,
            'tfidf': tfidf_features,
            'word2vec': w2v_features
        }
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    def clean_text(self, text):
        """Comprehensive text cleaning"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\b\d{10,11}\b', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove extra whitespace and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        cleaned = self.clean_text(text)
        lemmatized = self.tokenize_and_lemmatize(cleaned)
        return lemmatized
# Global variable to store the loaded pipeline
pipeline_data = None

def load_pipeline():
    """Load the saved spam detection pipeline"""
    global pipeline_data
    try:
        # Load the pipeline (adjust path if needed)
        pipeline_data = joblib.load('spam_detection_pipeline.pkl')
        print("Pipeline loaded successfully!")
        return True
    except FileNotFoundError:
        print("Pipeline file not found. Make sure 'spam_detection_pipeline.pkl' is in the same directory.")
        return False
    except Exception as e:
        print(f"Error loading pipeline: {str(e)}")
        return False

@app.route('/')
def home():
    """Home page with the form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if pipeline_data is None:
        return jsonify({
            'error': 'Pipeline not loaded. Please check if the model file exists.'
        }), 500
    
    try:
        # Get message from form
        message = request.form.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Please enter a message to analyze.'}), 400
        
        # Get the pipeline object
        pipeline = pipeline_data['pipeline']
        
        # Make prediction
        prediction = pipeline.predict([message])[0]
        probability = pipeline.predict_proba([message])[0]
        
        # Get probabilities for both classes
        ham_prob = probability[0]  # Probability of being ham (not spam)
        spam_prob = probability[1]  # Probability of being spam
        
        # Prepare response
        result = {
            'message': message,
            'prediction': prediction.upper(),
            'confidence': {
                'ham': round(ham_prob * 100, 2),
                'spam': round(spam_prob * 100, 2)
            },
            'is_spam': prediction == 'spam'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON input)"""
    if pipeline_data is None:
        return jsonify({
            'error': 'Pipeline not loaded. Please check if the model file exists.'
        }), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Please provide a message in JSON format.'}), 400
        
        message = data['message'].strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty.'}), 400
        
        # Get the pipeline object
        pipeline = pipeline_data['pipeline']
        
        # Make prediction
        prediction = pipeline.predict([message])[0]
        probability = pipeline.predict_proba([message])[0]
        
        # Prepare response
        result = {
            'message': message,
            'prediction': prediction,
            'probabilities': {
                'ham': round(probability[0], 4),
                'spam': round(probability[1], 4)
            },
            'is_spam': prediction == 'spam'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/model-info')
def model_info():
    """Display model information"""
    if pipeline_data is None:
        return jsonify({
            'error': 'Pipeline not loaded.'
        }), 500
    
    try:
        info = {
            'model_name': pipeline_data.get('model_name', 'Unknown'),
            'performance_metrics': pipeline_data.get('performance_metrics', {}),
            'preprocessing_info': pipeline_data.get('preprocessing_info', {})
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({
            'error': f'Failed to get model info: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Starting Spam Detection Flask App...")
    
    # Load the pipeline on startup
    if load_pipeline():
        print("Model Info:")
        if pipeline_data:
            print(f"   - Model: {pipeline_data.get('model_name', 'Unknown')}")
            metrics = pipeline_data.get('performance_metrics', {})
            if 'accuracy' in metrics:
                print(f"   - Accuracy: {metrics['accuracy']:.4f}")
        
        print("\nStarting Flask server...")
        print("   - Open your browser and go to: http://localhost:5000")
        print("   - Press Ctrl+C to stop the server")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print(" Failed to load pipeline. Please check the file path and try again.")