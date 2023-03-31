from flask import Flask, render_template, request, jsonify
import json

import classifier

app = Flask(__name__, template_folder='view')


@app.route('/')
def index():
    """serve homepage on this endpoint"""

    return render_template('index.html')

@app.route('/classify', methods = ['POST'])
def classify():
    """
    Classify Endpoint

    Keyword arguments:
    request -- http request
    Return: tweet classfication
    """
    tweet = json.loads(request.data)["tweet"]
    print(tweet)
    ret = classifier.classify(tweet)
    return jsonify(type=ret)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
