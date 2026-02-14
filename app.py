from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import re

app = Flask(__name__)

MODEL_PATH = "eeg_seizure_cnn_final.h5"
model = tf.keras.models.load_model(MODEL_PATH)

DATA_PATH = "dataset.csv"
df = pd.read_csv(DATA_PATH)
if 'Unnamed' in df.columns:
    df = df.drop(columns=['Unnamed'])
X = df.drop(columns=['y']).values
scaler = StandardScaler()
scaler.fit(X)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    error = None
    eeg_values = None

    if request.method == "POST":
        eeg_input = request.form.get("eeg_input")

        try:
            eeg_values = re.split(r'[,\s]+', eeg_input.strip())
            eeg_values = [float(x) for x in eeg_values if x]

            if len(eeg_values) != 178:
                error = f"You must enter exactly 178 EEG values! You entered {len(eeg_values)}."
            else:
                eeg_array = np.array(eeg_values).reshape(1, 178)
                eeg_scaled = scaler.transform(eeg_array)
                eeg_scaled = eeg_scaled.reshape(1, 178, 1)

                prob = model.predict(eeg_scaled)[0][0]
                probability = round(prob * 100, 2)
                result = "SEIZURE" if prob > 0.5 else "NON-SEIZURE"

        except ValueError:
            error = "Invalid input! Please enter numeric values separated by commas, spaces, or tabs."
            eeg_values = None

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        error=error,
        eeg_values=eeg_values,
        request=request
    )

if __name__ == "__main__":
    app.run(debug=True)
