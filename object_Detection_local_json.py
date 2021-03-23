from imageai.Detection import ObjectDetection
import os
import json

class object_Detection_local_json(object):
    def __init__(self):
        print("Initializing class .......")
        self.execution_path = os.getcwd()
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath( os.path.join(self.execution_path , "resnet50_coco_best_v2.1.0.h5"))
        #self.detector.setModelTypeAsTinyYOLOv3()
        #self.detector.setModelPath( os.path.join(self.execution_path , "yolo-tiny.h5"))
        self.detector.loadModel("fast")
        print("Loading model..........")

    def predict(self,X,feature_name):
        self.returned_image,self.detections = self.detector.detectObjectsFromImage(input_image=os.path.join(self.execution_path , X), minimum_percentage_probability=30, output_type='array')
        self.identity = dict()
        count = 1
        for eachObject in self.detections:
            self.identity[count]=[eachObject["name"],eachObject["percentage_probability"],eachObject["box_points"]]
            count += 1
        json_object = json.dumps(self.identity)
        return json_object