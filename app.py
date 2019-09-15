from flask import Flask, request
from model import trainAndGetPredictions
import json

app = Flask(__name__)

@app.route('/getPredictions')
def data():
    # here we want to get the value of income (i.e. ?income=some-value)
    income = request.args.get('income')
    age = request.args.get('age')
    return json.dumps(trainAndGetPredictions(income, age))

if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000
