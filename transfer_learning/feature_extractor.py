from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imutils import paths
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os

# Lay cac duong dan den anh
image_path = list(paths.list_images('dataset/'))

# Doi ngau nhien vi tri cac duong dan anh
random.shuffle(image_path)

labels = [path.split(os.path.sep)[-2] for path in image_path]

# Chuyen ten cac loai hoa thanh so
le = LabelEncoder()
labels = le.fit_transform(labels)

# Load model VGG16
model = VGG16(weights='imagenet', include_top=False)

# Load va resize anh
list_image = []
for (j, imagePath) in enumerate(image_path):
    image = load_img(imagePath, target_size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image, 0)
    image = imagenet_utils.preprocess_input(image)
    list_image.append(image)

list_image = np.vstack(list_image)

# Dung pre-trained model de lay ra cac feature cua anh
features = model.predict(list_image)
features = features.reshape((features.shape[0], 512*7*7))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

params = {'C' : [0.1 , 1.0, 10.0, 100.0]}
model = GridSearchCV(LogisticRegression(), params)
model.fit(X_train, y_train)
print('Best parameter for the model {}'.format(model.best_params_))

# Danh gia model
pred = model.presict(X_test)
print(classification_report(y_test, pred))











