# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS, cross_origin
# from cnnClassifier.utils.common import decodeImage, encodeImageIntoBase64
# from cnnClassifier.pipeline.predict import PredictionPipeline
# import os
# from pathlib import Path

# os.putenv('LANG', 'en_US.UTF-8')
# os.putenv('LC_ALL', 'en_US.UTF-8')

# app = Flask(__name__)
# CORS(app) 
     

# class ClientApp:
#     def __init__(self):
#         self.filename = "inputImage.jpg"
#         self.classifier = PredictionPipeline(self.filename)


# @app.route("/", methods=['GET'])
# @cross_origin()
# def home():
#     return render_template('index.html')


# @app.route("/train", methods=['GET', 'POST'])
# @cross_origin()
# def trainRoute():
#     os.system("python main.py")
#     return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
# def predictRoute():
#     image = request.json['image']
#     decodeImage(image, clApp.filename)
#     result = clApp.classifier.predict()
#     return jsonify(result)


# @app.route("/health", methods=['GET'])
# @cross_origin()
# def health():
#     return jsonify({"status": "healthy", "model": "VGG16 Chicken Disease Classifier"})


# if __name__ == "__main__":
#     clApp = ClientApp()
#     app.run(host='0.0.0.0', port=8080, debug=True)




# the Fast api code
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline
import os

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ImageRequest(BaseModel):
    image: str


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


@app.get("/")
def home():
    return {"message": "Chicken Disease Classifier API Running"}


@app.get("/train")
def train_route():
    os.system("python main.py")
    return {"message": "Training done successfully!"}


@app.post("/predict")
def predict_route(data: ImageRequest):
    decodeImage(data.image, clApp.filename)
    result = clApp.classifier.predict()
    return JSONResponse(content=result)


@app.get("/health")
def health():
    return {"status": "healthy", "model": "VGG16 Chicken Disease Classifier"}
