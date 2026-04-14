from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

model = tf.keras.models.load_model("disease_model.keras")

classes = ['Acne and Rosacea Photos', 'Bullous Disease Photos']

img_size = 256

def predict(model, img):
    img = img.resize((img_size, img_size))  # same size as training
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    return predicted_class, confidence

    img = img.resize
if __name__=="__main__":

    from PIL import Image

    img = Image.open("benign-familial-chronic-pemphigus-6.jpg")
    result = predict(model, img)

    print(result)

    