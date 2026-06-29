import cv2
from ultralytics import YOLO

class Occlusion:
    def __init__(self, face_model="app/models/yolov12n-face.pt", occlusion_model="app/models/classify.pt"):
        self.face_model = YOLO(face_model)
        self.occlusion_model = YOLO(occlusion_model)
        
    def predict(self, image):
        img = cv2.imread(image)

        if img is None:
            return {
                "status": "IMG_INVALID"
            }
        results = self.face_model.predict(image)

        if len(results[0].boxes) == 0:
            return {
                "status": "NO_FACE_DETECTED"
            }

        if len(results[0].boxes) > 1:
            return {
                "status": "MULTIPLE_FACES"
            }

        x1, y1, x2, y2 = results[0].boxes.xyxy.tolist()[0]
        cropped = img[int(y1):int(y2), int(x1):int(x2)]


        results = self.occlusion_model(cropped)
        top1 = results[0].probs.top1
        occluded = results[0].names[top1]

        return { "status": "SUCCESS",
                "occluded": occluded == "1"}