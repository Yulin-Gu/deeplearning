from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras import optimizers,losses,metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

#np.random.seed(7)
iris = load_iris()
X = iris['data']
Y = iris['target']
# 编码为0, 1, 2的对应的种类
#names = iris['target_names']
# 种类的名称, 字符串
#feature_names = iris['feature_names']
# 四个变量的名称

Y= iris.target.reshape(-1, 1) # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle = True, random_state=2)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))  
model.add(Dense(10, activation='relu'))   
model.add(Dense(3, activation='softmax'))  

model.summary()
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=200, batch_size=5, validation_data=(X_test,Y_test))

scores = model.evaluate(X, Y, 1)
print('scores:\n',scores)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.show()

y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)  #其实就是记录每个数组中值最大的数的index
y_test_class = np.argmax(Y_test, axis=1)

from sklearn.metrics import classification_report
report = classification_report(y_test_class, y_pred_class)
print(report)


