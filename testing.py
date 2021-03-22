from object_Detection_URL import object_Detection_URL
# image = "https://images.indianexpress.com/2020/11/delhi-metro-1.jpg"
image = "http://loyolaphoenix.com/wp-content/uploads/2020/09/image-1200x894.png"
# image = "https://pngimg.com/uploads/mario/mario_PNG53.png"
# image = "https://jpeg.org/images/jpegsystems-home.jpg"
# image = "http://jpeg-optimizer.com/images/sample/flower.jpg"
# image = "https://pngimg.com/uploads/scissors/scissors_PNG16.png"
#image = "https://upload.wikimedia.org/wikipedia/commons/4/4c/Vectorel-scissors-2245884.png"
#image = "https://cdn.store-factory.com/www.couteaux-services.com/content/product_9732713b.jpg?v=1518691523"
print(object_Detection_URL().predict(image,["feature_name"]))
