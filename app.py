from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    required = ["income", "credit_score", "loan_amount", "employment_years"]
    try:
        features = [[data[f] for f in required]]
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e.args[0]}"}), 400

    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max()

    decision = "Approved" if prediction == 1 else "Rejected"

    return jsonify({
        "loan_decision": decision,
        "confidence": round(float(confidence), 2)
    })

if __name__ == "__main__":
    app.run(port=5001)
