from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")  # your trained XGBoost model

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "No file uploaded", 400

        data = pd.read_csv(file)

        # Drop unnecessary columns if present
        for col in ["Time", "Amount", "Class"]:
            if col in data.columns:
                data = data.drop(col, axis=1)

        # Ensure columns match model expectations
        if data.shape[1] != 28:
            return "Uploaded data does not match expected number of features.", 400

        # Make predictions
        predictions = model.predict(data)
        data["Prediction"] = predictions

        frauds = data[data["Prediction"] == 1]

        if len(frauds) == 0:
            return render_template("index.html", message="No frauds detected!")

        return render_template("index.html",
                               tables=[frauds.to_html(classes='data', header="true", index=False)],
                               fraud_count=len(frauds),
                               message=f"Detected {len(frauds)} fraudulent transactions.")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
