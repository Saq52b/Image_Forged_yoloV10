from flask import Flask, request, jsonify
import cv2
import supervision as sv
from ultralytics import YOLOv10
import base64
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route("/test1", methods=['POST'])
def hello_world():
    model = YOLOv10('./runs/detect/train5/weights/best.pt')
    image = cv2.imread('D:/Saqib/IdentifyModifiedImage/DS/TEMP_IMG/v30.png')
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)
    return detections.data['class_name'].tolist()[1]
@app.route("/GET_PREDICTION_DATA", methods=['POST'])
def test():
    try:
        app.logger.info("Received POST request")
        data = request.get_json()
        if not data:
            app.logger.error("Invalid or missing JSON body")
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        if 'base64String' not in data:
            app.logger.error("Missing base64String")
            return jsonify({"error": "Missing base64String"}), 400

        base64_string = data['base64String']
        app.logger.info(f"Received base64 string of length {len(base64_string)}")

        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image1 = Image.open(BytesIO(image_data))
        path = './runs/detect/train5/weights/last.pt'
        model = YOLOv10(path)
        results = model(image)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Annotate the image
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        _, buffer = cv2.imencode('.png', annotated_image)
        encoded_string = base64.b64encode(buffer).decode('utf-8')
        # GET_RESULT_STRING(image,model)
        app.logger.info("Image processed and encoded successfully")
        return jsonify({"image": encoded_string,"detection_result":detections.data['class_name'].tolist()[1]})

    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run( debug=True)
