from flask import Flask,render_template,request
import os
import cv2
import numpy
from keras.models import load_model
app=Flask(__name__)

def processImg(filepath):
    model=load_model("mask.model")
    print("hello")
    img_size = 300
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)/255  
    img_resize = cv2.resize(img, (img_size, img_size))  
    last=img_resize.reshape(-1, img_size, img_size, 1)

    prediction = model.predict(last)
    print((prediction))

    CATEGORIES = ["withoutMask", "withMtask"]

    pred_class = CATEGORIES[numpy.argmax(prediction)]
    print(pred_class)
    return pred_class



@app.route('/')
def index():
    return render_template("index.html")

@app.route("/prediction",methods=["POST"])
def prediction():
    img=request.files['myfile']
    img.save("img.jpg")
    resp = processImg("img.jpg")
    return render_template("res.html" , result=resp)

if __name__=="__main__":
    app.run(debug=True)