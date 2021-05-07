import os
from flask import Flask, render_template
from flask import request
import boto3
import json
import requests

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        try:
            txt = request.form['url']
            print(txt)
        except:
            errors.append(
                "Unable to get a twit. Please make sure it's valid and try again."
            )
        try:
            url = 'https://bbs4eeufai.execute-api.eu-central-1.amazonaws.com/v1/predict-twit-lier'
            response = requests.post(url, data = json.dumps({"data":txt}))
            result = [float(i) for i in str(response.content)[3:-2].split(', ')]
            results = [['Probability Fake', result[0]], ['Probability True', result[1]]]

        except:
            errors.append("Unable to connec to sagemaker endpoint.")
    return render_template('index.html', errors=errors, results=results)


if __name__ == '__main__':
    app.run()
