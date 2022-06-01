# -*- coding: utf-8 -*


###Spyder Editor
#import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB2,EfficientNetB6
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras

# %%%
#create labels
classes=[]
filename=r"C:\\Users\diego\OneDrive\Desktop\DS\brain_cancer\Brain_cancer_classification\data"
for sub_folder in os.listdir(os.path.join(filename,'Training')):
    classes.append(sub_folder)
print(classes)

# %%%
#resize images and put together Training and Testing folder
X_train = []
y_train = []
image_size = 160
for i in classes:
    path_train = os.path.join(filename,'Training',i)
    for j in tqdm(os.listdir(path_train)): #Instantly make your loops show a smart progress meter 
        img = cv2.imread(os.path.join(path_train,j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)
    path_test = os.path.join(filename,'Testing',i)
    for j in tqdm(os.listdir(path_test)):
        img = cv2.imread(os.path.join(path_test,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)
        
X_train = np.array(X_train)
y_train = np.array(y_train)        


#%%%

#data augmentation
X_train, y_train = shuffle(X_train,y_train, random_state=42)
datagen = ImageDataGenerator(
    rotation_range=7,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True)

datagen.fit(X_train)
X_train.shape
lb = LabelEncoder()

#train and test splitting 
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.15,random_state=42,stratify=y_train)

labels_train=lb.fit(y_train)
y_train=lb.transform(y_train)
y_test=lb.transform(y_test)

#%%%

print(y_train)
#%%%

#load EfficientNet
EfficientNet=EfficientNetB6(weights='imagenet', include_top=False,input_shape=(image_size,image_size,3))

#%%%
#train the model
tf.random.set_seed(79)
model = EfficientNet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4,activation='softmax')(model)
model = tf.keras.models.Model(inputs=EfficientNet.input, outputs = model)
opt = Adam(
    learning_rate=0.000023,
    epsilon=1e-08,
    clipnorm=1.0)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# summarize the model
print(model.summary())
# fit the model
early_stopping_cb=keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)


history=model.fit(X_train ,y_train,validation_data = (X_test,y_test),epochs=70,
    batch_size=8,callbacks=[early_stopping_cb])

#%%%
# change directory
os.chdir(r'C:\\Users\diego\OneDrive\Desktop\DS\brain_cancer\Brain_cancer_classification')
print(os.getcwd())

#save the model
model.save(os.path.join('models/','EfficientNetB6.h5'))
model.save_weights(os.path.join('models/','EfficientNetB6_weights.h5'))

#plot loss and accuracy
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)

#plt.gca().set_xlim(0,33)
plt.gca().set_ylim(0,1)
plt.savefig(os.path.join('plots/','EfficientNetB6.png'), dpi=500)
loss, accuracy = model.evaluate(X_test,y_test)

#print accuracy    
print('Accuracy: %f' % (accuracy*100))

#%%%

#load_weights 
#image_size = 160 
#EfficientNet=EfficientNetB6(weights='imagenet', include_top=False,input_shape=(image_size,image_size,3))
#model = EfficientNet.output
#model = tf.keras.layers.GlobalAveragePooling2D()(model)
#model = tf.keras.layers.Dropout(rate=0.5)(model)
#model = tf.keras.layers.Dense(4,activation='softmax')(model)
#model = tf.keras.models.Model(inputs=EfficientNet.input, outputs = model)




#%%%
  #load the model
model=keras.models.load_model(os.path.join('models/','EfficientNetB6.h5'))   
#  More details about the model
model.summary()
loss, accuracy = model.evaluate(X_test,y_test)

#print accuracy    
#print('Accuracy: %f' % (accuracy*100))
# visualize activation functions
#for i, layer in enumerate (model.layers):
#    print (i, layer)
#    try:
#        print ("    ",layer.activation)
#    except AttributeError:
#        print('   no activation attribute')
#specific info about each layer
#for i in range(len(model.layers)):
#    print(f'{i}   {model.layers[i]}: \n{model.layers[i].get_config()} \n')
#info about optimizers
#model.optimizer.get_config()    

#%%%







