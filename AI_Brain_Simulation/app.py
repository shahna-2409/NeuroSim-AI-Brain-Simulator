from flask import render_template, Flask, request, jsonify, redirect
from flask_cors import CORS, cross_origin
import os
from predict import accient

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)
@app.route("/", methods=["GET", "POST"])
def login():
    
    if request.method == "POST":
        first_name = request.form.get("username")
        last_name = request.form.get("password") 
        if first_name == 'admin' and last_name == 'admin':
            return redirect("/home")
    else:
        return render_template("login.html")

@app.route("/home", methods=['GET'])
@cross_origin()
def home():
    if request.method == "GET":
        return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
   if request.method == "POST":
    image_file = request.files['file']
    print(image_file)
    classifier = accient()  
    result = classifier.accientImage(image_file)
    return result
   else:
       print('Loading Error')


if __name__ == "__main__":
    app.run(debug=True)

