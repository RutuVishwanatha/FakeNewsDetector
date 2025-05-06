from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news']
    vector_input = vectorizer.transform([input_text])
    prediction = model.predict(vector_input)
    result = "Fake News" if prediction[0] == 'FAKE' else "Real News"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
