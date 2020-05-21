from flask import Flask, jsonify
import numpy as np
import json
import requests
from flask import render_template

app = Flask(__name__)

# Define what to do when a user a specific route
@app.route("/")
def index1():
    return render_template("index.html")

@app.route("/data.html")
def data():
    return render_template("data.html")

@app.route("/about.html")
def about():
    # print("Server received request for 'Home' page...")
     return render_template("about.html")

@app.route("/license.html")
def license():
    return render_template("license.html")

@app.route("/index.html")
def index2():  
    return render_template("index.html")

# run app
if __name__ == "__main__":
    app.run(debug=True)