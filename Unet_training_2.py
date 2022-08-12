# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 23:10:46 2022

@author: Abel
"""

from Unet_jaccard import simple_unet_model_with_jacard    #Use normal unet model
#from simple_unet_model import simple_unet_model   #Use normal unet model

from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#%%
#%%

image_directory = 'dataset/images/'
mask_directory = 'dataset/labels/'
#%%
SIZE = 128
image_dataset = []  
mask_dataset = []  
#%%
images = os.listdir(image_directory)
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'jpg'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, -1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))
#%%
masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))
#%%
print(np.shape(image_dataset))
print(np.shape(mask_dataset))
#%%
#Normalize images
image_dataset=np.array(image_dataset).astype('float32')/255.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.
#%%
print(np.shape(image_dataset))
print(np.shape(mask_dataset))
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

#X_train_quick_test, X_test_quick_test, y_train_quick_test, y_test_quick_test = train_test_split(X_train, y_train, test_size = 0.9, random_state = 0)
#%%
#X_train=np.load("X_train_ISIC2018.npy")
#y_train=np.load("y_train_ISIC2018.npy")
#X_test=np.load("X_test_ISIC2018.npy")
#y_test=np.load("y_test_ISIC2018.npy")
#

#%%
#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(X_train[image_number], cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(y_train[image_number], cmap='gray')
plt.show()

#%%
###############################################################
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

#%%
def get_jacard_model():
    return simple_unet_model_with_jacard(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model_jacard = get_jacard_model()


history_jacard = model_jacard.fit(X_train, y_train, 
                    batch_size = 32, 
                    verbose=1, 
                    epochs=15, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)




model_jacard.save('Unet_with_jacard.hdf5')




#%%
# evaluate model
_, jaccard, acc = model_jacard.evaluate(X_test, y_test)
print("Accuracy of Jacard Model is = ", (jaccard * 100.0), "%")
#%%
#%%
#plot the training and validation accuracy and loss at each epoch
loss = history_jacard.history['loss']
val_loss = history_jacard.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

jc = history_jacard.history['dice_coef']
#acc = history.history['accuracy']
val_jc = history_jacard.history['val_dice_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, jc, 'y', label='Training Jacard Coeff.')
plt.plot(epochs, val_jc, 'r', label='Validation Jacard Coeff.')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Jacard Coefficient')
plt.legend()
plt.show()
#%%
from tensorflow.keras.metrics import MeanIoU

predictions = (model_jacard.predict(X_test, batch_size=8)[:,:,:] > 0.5)#.astype(np.uint8)
IoU = MeanIoU(num_classes=2)
IoU.update_state(y_test, predictions)
mean_IoU = IoU.result().numpy()
print("Mean IoU for testing dataset is: ", mean_IoU)
#%%
from sklearn.metrics import f1_score, precision_score, recall_score
y_test_flatten=np.concatenate(y_test).flatten().astype(np.uint8)
predictions_flatten=np.concatenate(predictions).flatten().astype(np.uint8)
f1_score = f1_score(y_test_flatten, predictions_flatten)
recall_result = recall_score(y_test_flatten, predictions_flatten)
precision_result = precision_score(y_test_flatten, predictions_flatten)
print("F-1 Score for testing dataset is: ", f1_score)
print("Recall for testing dataset is: ", recall_result)
print("Precision Score for testing dataset is: ", precision_result)

#%%
import random
test_img_number = random.randint(0, X_test.shape[0]-1)  #Test with 119

test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]

#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model_jacard.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()
#%%
import random
test_img_number = 11  #Test with 119

test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]

#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model_jacard.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()

#%%
import random
test_img_number = 64  #Test with 119

test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]

#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model_jacard.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()


#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%












