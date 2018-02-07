from flask import Flask, request, redirect, jsonify
import os
from werkzeug.utils import secure_filename
from sys import argv
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "/CapstoneWithUI-prototype/upload"


@app.route("/")
def index():
    return redirect("/static/index.html")


@app.route("/sendfile", methods=["POST"])
def send_file():
    fileob = request.files["file2upload"]
    filename = secure_filename(fileob.filename)
    save_path = "{}/{}".format(app.config["UPLOAD_FOLDER"], filename)
    fileob.save(save_path)
    isSignificant = 0.8 #try different values.

    # P: list of probabilities
    Result, P, classNames = aT.fileClassification(save_path, "svmModel", "svm")
    winner = np.argmax(P) #pick the result with the highest probability value.

    # is the highest value found above the isSignificant threshhold?
    if P[winner] > isSignificant :
        result = "File: " +filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner])
        print("File: " +filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner]))
        with open("Result.txt", "a") as myfile:
            myfile.write(result+"\n")
    else :
        print("Probability MAP: " + str(P))
        result = "File"+filename+" is in Category: " + classNames[winner] + ", Probability: "+ str(P[winner])
        with open("Result.txt", "a") as myfile:
            myfile.write(result + "\n")
        print("Category: " + classNames[winner] + ", Probability: "+ str(P[winner]))
    

    #open and close to update the access time.
    with open(save_path, "r") as f:
        pass

    return "successful_upload"


@app.route("/filenames", methods=["GET"])
def get_filenames():
    filenames = os.listdir("uploads/")

    #modify_time_sort = lambda f: os.stat("uploads/{}".format(f)).st_atime

    def modify_time_sort(file_name):
        file_path = "uploads/{}".format(file_name)
        file_stats = os.stat(file_path)
        last_access_time = file_stats.st_atime
        return last_access_time

    filenames = sorted(filenames, key=modify_time_sort)
    return_dict = dict(filenames=filenames)
    return jsonify(return_dict)


if __name__ == '__main__':
    app.run(debug=False)
