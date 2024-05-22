import time
from pathlib import Path

import pymongo
from flask import Flask, request
import uuid
import yaml
from loguru import logger
import os
import torch
import boto3
import yolo5.oldapp


images_bucket = os.environ['BUCKET_NAME']

# mongo_uri = os.environ['MONGO_URI']
# client = pymongo.MongoClient(mongo_uri)
# db = client['predictions']
# ---- added mongo_uri, 3 rows ----

# with open("data/coco128.yaml", "r") as stream:
#    names = yaml.safe_load(stream)['names']
# I commented 'class': names[int(l[0])],

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict() -> str:
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())

    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    # img_name = request.args.get('imgName')
    downloaded_image_path = '/home/shantalberkhof/PycharmProjects/DockerProject/local_file.jpg'
    s3_image_path = 'data/file_5.jpg'

    # TODO download img_name from S3, store the local image path in the original_img_path variable.
    #  The bucket name is provided as an env var BUCKET_NAME.
    original_img_path = '/photos/street.jpg'

    s3 = boto3.client('s3')
    local_img_path = f'{downloaded_image_path}'
    original_img_path = downloaded_image_path
    print("before download")
    s3.download_file(images_bucket, s3_image_path, downloaded_image_path)
    print("after download")

    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
    # Model loading
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Can be 'yolov5n' - 'yolov5x6', or 'custom'

    # Inference on images
    # original_img_path = "https://ultralytics.com/images/zidane.jpg"  # Can be a file, Path, PIL, OpenCV, numpy, or list of images

    # Run inference
    results = model(original_img_path)

    # results.save()  # Save images

    # Display results
    #print("we should see results now")
    #results.print()
    #results_path = "/home/shantalberkhof/PycharmProjects/DockerProject/results.txt"
    #results.save(results_path)  # Other options: .show(), .save(), .crop(), .pandas(), etc.
    # Summery = results.xyxy[0]
    # Summery = results.print
    #json_data = results.pandas().xyxy[0].to_json(orient="records")
    #print(Summery)
    #print(json_data)  # JSON img1 predictions

    # Predicts the objects in the image
    # yolo5.app.predict.run(
    # yolo5.app.predict(

    #    weights='yolov5s.pt',

    #    data='data/coco128.yaml',
    #    source=original_img_path,
    #    project='static/data',
    #    name=prediction_id,
    #    save_txt=True
    # )
    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')
    #return (Summery)

    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).

    #s3.upload_file(local_img_path, images_bucket, f'predicted/{downloaded_image_path}')
    s3.upload_file(local_img_path, images_bucket, f'{s3_image_path}.predicted')
    # ---- added ----
    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
    if pred_summary_path.exists():
        print("checkpoint 1")
        with open(pred_summary_path) as f:
            print("checkpoint 2")
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
#                'class': names[int(l[0])],

                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path,
            'labels': labels,
            'time': time.time()
        }
        print("printing summary")
        print(prediction_summary)
        # TODO store the prediction_summary in MongoDB

        # db.predictions.insert_one(prediction_summary)

        # ----- added ----

        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
