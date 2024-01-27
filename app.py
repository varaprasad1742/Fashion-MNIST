from flask import Flask,url_for,render_template,request
import pickle
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np

gpu_memory_fraction = 0.5

# Configure TensorFlow to limit GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to allocate only a fraction of GPU memory
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(gpu_memory_fraction * 1024))]
        )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)




app=Flask(__name__)
app.config['UPLOAD_FOLDER']='uploads'

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Load the model, providing the custom_objects parameter
model = load_model('final-last.h5')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    file=request.files
    if 'file' not in file:
        return 'No file part'
    file=file['file']
    if file.filename == '':
        return 'Upload failure'
    path=f'./static/{file.filename}'
    file.save(path)
    img=cv2.imread(path,0)
    img=cv2.resize(img,(28,28))
    img=img/255.0
    img=img.reshape(1,28,28,1)
    pred=model.predict(img)
    pred=np.argmax(pred)
    pred=class_names[pred]
    print(pred)
    return render_template('index.html',imagePath=file.filename,result=pred)

if __name__=='__main__':
    app.run(debug=True)