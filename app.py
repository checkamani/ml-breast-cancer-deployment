import os
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model (update filename when you save your model)
MODEL_PATH = "model.pkl"
model = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Model not loaded yet: {e}")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["Clump_thickness"]),
            float(request.form["Uniformity_of_cell_size"]),
            float(request.form["Uniformity_of_cell_shape"]),
            float(request.form["Marginal_adhesion"]),
            float(request.form["Single_epithelial_cell_size"]),
            float(request.form["Bare_nuclei"]),
            float(request.form["Bland_chromatin"]),
            float(request.form["Normal_nucleoli"]),
            float(request.form["Mitoses"]),
        ]

        pred = model.predict([features])[0]
        return render_template("index.html", prediction=int(pred), error=None)

    except Exception as e:
        return render_template("index.html", prediction=None, error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
