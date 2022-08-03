import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
import  matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
import pandas as pd
import os


batch_size = 32
num_classes = 2
epochs = 200
(img_height, img_width) = (128,128)
input_shape = (img_height, img_width, 3)

if os.name == 'nt':
    # 即windows系统
    seperator = '\\'
else:
    # 即unix或者Linux系统
    seperator = '/'


train_data_dir = '.{}catsdogs{}train'.format(seperator,seperator)
validation_data_dir = '.{}catsdogs{}valid'.format(seperator,seperator)

'''
train_data_dir = os.path.abspath(os.path.join(os.getcwd(), "../catsdogs/train"))
validation_data_dir = os.path.abspath(os.path.join(os.getcwd(), "../catsdogs/valid"))
'''

vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=input_shape)
vgg.summary()


output = vgg.layers[-1].output

output = tf.keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = True

set_trainable = False
for layer in vgg_model.layers:
#    if layer.name in ['block5_conv1', 'block4_conv1']:
    if layer.name in ['block5_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
#pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])      
#pd.set_option('max_colwidth', -1)

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape, kernel_regularizer =tf.keras.regularizers.l2(l=0.00005)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu', kernel_regularizer =tf.keras.regularizers.l2(l=0.00005)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-6),
              metrics=['accuracy'])

model.summary()


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   zoom_range=0.3, 
                                   rotation_range=50,
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   horizontal_flip=True, 
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

history = model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=STEP_SIZE_VALID)

#model.save('cats_dogs_tlearn_finetune_img_aug_cnn.h5')


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Transfer Learning with Data Aug and Fine Tune CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.show()