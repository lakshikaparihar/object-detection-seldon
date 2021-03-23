#################################################################################
# URL  returning  IMAGE
#################################################################################
from object_Detection_URL_image  import object_Detection_URL_image

url = ""
print(object_Detection_URL_image().predict(url,"feature_name"))

#################################################################################
# URL  returning  JSON
#################################################################################

from object_Detection_URL_json import object_Detection_URL_json

url = ""
print(object_Detection_URL_json().predict(url,"feature_name"))

################################################################################
# LOCAL  returning  IMAGE
#################################################################################

from object_Detection_local_image  import object_Detection_local_image
img2 = "images\img.png"
print(object_Detection_local_image().predict(img2,["feature_name"]))

################################################################################
# LOCAL  returning  JSON
#################################################################################

from object_Detection_local_json  import object_Detection_local_json
img2 = "images\img.png"
print(object_Detection_local_json().predict(img2,["feature_name"]))