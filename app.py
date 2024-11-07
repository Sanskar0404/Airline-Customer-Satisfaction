import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__, template_folder="template", static_folder="staticfiles")
model = pickle.load(open('build.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form and convert them to float
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Predict and interpret the result
    prediction = model.predict(final_features)
    output = prediction[0]
    
    # Display satisfaction result based on the prediction
    if output == 1:
        satisfaction_text = "Customer is Satisfied"
    else:
        satisfaction_text = "Customer is Not Satisfied"
    
    return render_template('index.html', prediction_text=satisfaction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
