FROM seldonio/seldon-core-s2i-python3:1.6.0
# ADD requirements.txt .
# RUN pip install -r requirements.txt

RUN pip install tensorflow==2.4.0
# opencv-python==4.2.0.34
# keras==2.4.3 
# pillow==7.0.0 
# scipy==1.4.1 
# h5py==2.10.0
# imageai
