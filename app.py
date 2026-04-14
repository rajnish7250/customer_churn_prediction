from flask import Flask, request, jsonify
from predict import predict

app = Flask(__name__)

@app.route('/')
def home():
    return "Churn Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict_churn():
    data = request.json['features']
    result = predict(data)
    return jsonify({'churn': int(result)})

if __name__ == '__main__':
    app.run(debug=True)