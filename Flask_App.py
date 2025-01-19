from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

model = joblib.load('model_tfidf.pkl')
vectorizer = joblib.load('vectorizer.pkl')

#Text preprocessing

def remove_special_characters(text):
  text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
  return text


def remove_stopwords(text):
    """Removes stop words from a string."""
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)


def preprocess_text(text):
    text = text.lower()
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# creating a rendering route to our html file
@app.route('/')
def home():
    return render_template('index.html')

# reacting to submit button on html file, redirecting to /predict, then returning the model's prediction on the
# user review
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request.form)  
        user_input = request.form['review']
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()
        prediction = model.predict(vectorized_text)
        sentiment = "Pozitív vélemény/Positive review" if prediction > 0.5 else "Negatív vélemény/negative review"
        return render_template('index.html', prediction=sentiment)


if __name__ == '__main__':
    app.run(debug=True)