from gettext import install
import numpy as np
import joblib
from flask import Flask, request, render_template



app1= Flask(__name__)
model = joblib.load(open('model_4.pkl', 'rb'))
cv = joblib.load(open('cv_4.pkl', 'rb'))


@app1.route('/')
def home():
    return render_template('index1.html')


@app1.route('/predict', methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    # if request.method == 'POST':
    text = request.form.get('Review')
    data = [text]
    vectorizer = cv.transform(data).toarray()
    prediction = int(model.predict(vectorizer))
    if prediction==1:
        return render_template('index1.html', predicted_review=text,prediction_text='The review is Positive')
    else:
        return render_template('index1.html', predicted_review= text, prediction_text='The review is Negative.')


if __name__ == "__main__":
    app1.run(debug=True)