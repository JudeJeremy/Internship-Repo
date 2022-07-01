from flask import Flask, render_template, jsonify, request
import os

app = Flask(__name__)

def addit(num1, num2):
    return str(int(num1) + int(num2))

@app.route('/', methods=['GET'])
def add_numbers():
    num1 = 200
    num2 = 300
    op = addit(num1, num2)
    return render_template('addNumbers.html', sum=op)
   
    

    




if __name__ == "__main__":
    app.run(debug=True)