from imageai.Detection import VideoObjectDetection
import os
import json

class object_Detection_video(object):
    def __init__(self):
        print("Initializing class .......")
        self.execution_path = os.getcwd()
        self.detector = VideoObjectDetection()
        #self.detector.setModelTypeAsRetinaNet()
        #self.detector.setModelPath( os.path.join(self.execution_path , "resnet50_coco_best_v2.1.0.h5"))
        self.detector.setModelTypeAsTinyYOLOv3()
        self.detector.setModelPath( os.path.join(self.execution_path , "yolo-tiny.h5"))
        self.detector.loadModel()
        print("Loading model..........")

    def predict(self,feature_name):
        self.detections = self.detector.detectObjectsFromVideo(input_file_path=os.path.join( self.execution_path, "traffic-mini.mp4"),output_file_path=os.path.join(self.execution_path, "traffic_mini_detected_1"), frames_per_second=29, log_progress=True)
        return self.detections