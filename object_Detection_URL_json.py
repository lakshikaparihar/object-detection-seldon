from imageai.Detection import ObjectDetection
import os
import json
from skimage import io
from numpy import asarray
from PIL import Image
import requests

class object_Detection_URL_json(object):
    def __init__(self):
        print("Initializing class .......")
        self.execution_path = os.getcwd()
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath( os.path.join(self.execution_path , "resnet50_coco_best_v2.1.0.h5"))
        self.detector.loadModel("fast")
        print("Loading model..........")

    def predict(self,X,feature_name):
        print("Entering.....................................")
        self.im = Image.open(requests.get(X, stream=True).raw)
        self.im.save("image.png")
        self.im_f = "image.png"
        self.image_numpy = io.imread(self.im_f)
        self.image_numpy = self.image_numpy[:,:,:3]
        print("PNG is ready to detect..................")
        print("Detecting........................................")
        self.returned_image,self.detections= self.detector.detectObjectsFromImage(input_type="array", input_image=self.image_numpy , minimum_percentage_probability=30, output_type='array')
        self.identity = dict()
        count = 1
        for eachObject in self.detections:
            self.identity[count]=[eachObject["name"],eachObject["percentage_probability"],eachObject["box_points"]]
            count += 1
        json_object = json.dumps(self.identity)
        return json_object