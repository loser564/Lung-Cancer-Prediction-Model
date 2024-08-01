# use model to predict
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
model = load_model('lung_cancer_model.h5')

# test image
img = Image.open('lung_image_sets/lung_aca/lungaca1.jpeg')
img = img.resize((224, 224))
img = np.array(img)
img = np.expand_dims(img, axis=0)

# predict
prediction = model.predict(img)
# print accuracy
print(prediction)
# print class
print(np.argmax(prediction))
# get accuracy
print(np.max(prediction))