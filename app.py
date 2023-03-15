from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    if request.method == 'POST':
        print("Hello")
        num = request.form['number']
        # try:
    
        output = model.predict([[float(num)]])
        output = output
        # except ValueError:
        #     output = 'Invalid input!'
    return render_template('index.html', output=output)

# def predict(x):
#     # Get the input data from the POST request
#     # input_data = request.get_json()
#     # x = np.array([input_data['col1']])

#     # Make a prediction using the model
#     y_pred = model.predict(x.reshape(1, -1))

#     # Return the prediction as a JSON response
#     response = {'prediction': float(y_pred)}
#     # return jsonify(response)
#     return y_pred

if __name__ == '__main__':
    app.run(debug=True)