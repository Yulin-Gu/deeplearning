import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses,activations,optimizers
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn import metrics

import warnings
from keras.models import Model
from keras.layers import Input,Activation,Dropout,Reshape,Conv2D,MaxPooling2D,Dense,Flatten
from keras import backend as K

print('cwd=',os.getcwd())

if os.name == 'nt':
    # 即windows系统
    seperator = '\\'
else:
    # 即unix或者Linux系统
    seperator = '/'

train_data_dir = '.{}catsdogs{}train'.format(seperator,seperator)
validation_data_dir = '.{}catsdogs{}valid'.format(seperator,seperator)
test_data_dir = '.{}catsdogs{}test'.format(seperator,seperator)

def resize_image(img, size):
    
    if np.array(img).shape[2] == 4:
        img = img.convert('RGB')
    
    img.thumbnail(size, Image.ANTIALIAS)
    newimg = Image.new("RGB", size, (255, 255, 255))
    newimg.paste(img, (int((size[0] - img.size[0]) / 2), int((size[1] - img.size[1]) / 2)))
    return newimg


train_datagen = ImageDataGenerator(
    rescale=1./255,   # normalization
    shear_range=0.2,
    zoom_range=0.2,   
    horizontal_flip=True)  

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(128, 128),
    batch_size=1,
    shuffle=False,
    class_mode='categorical')

classnames = np.array(['cat','dog'])
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
print('STEP_SIZE_TRAIN =',STEP_SIZE_TRAIN)
print('STEP_SIZE_VALID =',STEP_SIZE_VALID)
print('STEP_SIZE_TEST =',STEP_SIZE_TEST)


model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape=(128,128,3),activation='relu'))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Flatten())
model.add(Dense(len(classnames), activation='softmax'))

model.summary()
#就是输出的那个表
optimizer = optimizers.Adam
loss = losses.categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

num_epochs = 50

history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID)

print(history)

# plot training accuracy and loss from history
plt.figure(figsize=(15,10))
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.show()

# load predict images
predict_data = []
predict_images = []
size = (128,128) 

files = os.listdir(test_data_dir)
for f in files:

    files2 = os.listdir(os.path.join(test_data_dir,f))
    for f2 in files2: 
    #img = Image.open(os.path.join('C:/Users/DAWN/Desktop/CNN/catsdogs/test/',f))
        img = Image.open(os.path.join(test_data_dir,f,f2))
        img = resize_image(img,size)
        predict_data.append(np.array(img))

# 猫是0, 狗是1


predict_data = np.array(predict_data,dtype='float')/255.

predicted_labels_encoded = model.predict(predict_data)
print('predicted_labels_encoded =',predicted_labels_encoded)
# predicted_labels_encoded is an array ,so np.argmax must specify the axis, so the result is also an arrar                                 
predicted_labels = np.argmax(predicted_labels_encoded,axis=1) 
print('predicted digits={}'.format(predicted_labels))                                
#predicted_labels = encoder.inverse_transform(predicted_labels)
#print('predicted labels=',predicted_labels)

predicted_labels_encoded2 = model.predict(test_generator,steps=STEP_SIZE_TEST)
print('predicted_labels_encoded2 =',predicted_labels_encoded2)
predicted_labels2 = np.argmax(predicted_labels_encoded2,axis=1) 
print('predicted digits2={}'.format(predicted_labels2)) 

y_pred = predicted_labels
y_true = predicted_labels2

# print acc, precision, recall, f1_score, table
accuracy = metrics.accuracy_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred, average="macro")
precision = metrics.precision_score(y_true, y_pred, average="macro")
F1 = metrics.f1_score(y_true, y_pred, average="macro")  
t = classification_report(y_true, y_pred, target_names=['0','1'])
print(" accuracy:",  accuracy, '\n', "precision:", precision, '\n', "recall:", recall, '\n', "F1:", F1,'\n')
print(t)