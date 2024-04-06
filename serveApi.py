from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from pydantic import BaseModel
app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    return jsonify({"message": "Hi, welcome to the bike prediction API!"})

if __name__ == "__main__":
    app.run(debug=True)
