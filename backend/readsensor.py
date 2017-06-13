from flask import Flask, jsonify
from flask import request
app = Flask(__name__)

data = []

@app.route('/input')
def api_input():
    if 'input' in request.args:
    	data.append(request.args['input'])
        return 'Wrote ' + request.args['input']
    else:
        return 'Put data into input parameter'

@app.route("/fetch")
def get():
    return jsonify(results=data)

@app.route('/hello')
def api_hello():
    if 'name' in request.args:
        return 'Hello ' + request.args['name']
    else:
        return 'Hello John Doe'

if __name__ == "__main__":	
    app.run()