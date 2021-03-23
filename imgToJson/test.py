from Model import Model
img = 'D:\Projects\object-detection-seldon\imgToJson\AIM2.png'
print(Model().predict(img,["feature_name"]))