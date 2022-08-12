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

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy
from loss_functions import *

beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)  # -1 ultiplied as we want to minimize this value as loss function


def dice_coef(y_true, y_pred, smooth=100):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def sensitivity(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
        true_negatives = K.sum(
            K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

def weighted_cross_entropyloss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        targets=y_true,
                                                        pos_weight=pos_weight)
        return tf.reduce_mean(loss)

def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = focal_loss_with_logits(logits=logits, targets=y_true,
                                      alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.reduce_mean(loss)

def depth_softmax(matrix):
        sigmoid = lambda x: 1 / (1 + K.exp(-x))
        sigmoided_matrix = sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
        return softmax_matrix

def generalized_dice_coefficient(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

def dice_loss(y_true, y_pred):
        loss = 1 - generalized_dice_coefficient(y_true, y_pred)
        return loss

def bce_dice_loss(y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
        return loss / 2.0

def bce_dice_loss2(y_true, y_pred,a,b):
        loss = a*binary_crossentropy(y_true, y_pred) + b*dice_loss(y_true, y_pred)
        return loss / 2.0


def confusion(y_true, y_pred):
        smooth = 1
        y_pred_pos = K.clip(y_pred, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.clip(y_true, 0, 1)
        y_neg = 1 - y_pos
        tp = K.sum(y_pos * y_pred_pos)
        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)
        prec = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        return prec, recall

def true_positive(y_true, y_pred):
        smooth = 1
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pos = K.round(K.clip(y_true, 0, 1))
        tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
        return tp

def true_negative(y_true, y_pred):
        smooth = 1
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos
        tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
        return tn

def tversky_index(y_true, y_pred):
        smooth = 1    
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                    1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
        return 1 - tversky_index(y_true, y_pred)

def focal_tversky(y_true, y_pred):
        pt_1 = tversky_index(y_true, y_pred)
        gamma = 0.75
        return K.pow((1 - pt_1), gamma)

def log_cosh_dice_loss(y_true, y_pred):
        x = dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)



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
#%%
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
s = Semantic_loss_functions(1,1)

# create model
model = KerasClassifier(model=get_jacard_model, loss=s.bce_dice_loss2, optimizer="adam", epochs=15, batch_size=2, verbose=1)

# define the grid search parameters
a_values = [0.1, 0.2, 0.3]
b_values = [0.2,0.4, 0.5]
param_grid = dict(loss__a=a_values, loss__b=b_values)

## create model
#model = KerasClassifier(model=get_jacard_model, loss="binary_crossentropy", optimizer="SGD", epochs=15, batch_size=10, verbose=1)
## define the grid search parameters
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#param_grid = dict(optimizer__learning_rate=learn_rate, optimizer__momentum=momentum)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None, pre_dispatch = 2,cv=2)
xnsamples, xx, xy, xz = X_train.shape
ynsamples, yx, yy, yz = y_train.shape

x_train_dataset = X_train.reshape((xnsamples,xx*xy*xz))
y_train_dataset = y_train.reshape((ynsamples,yx*yy*yz)).astype(np.uint8)
grid_result = grid.fit(x_train_dataset, y_train_dataset)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))







#%%
model_jacard.compile(optimizer = 'adam', loss = [focal_loss_with_logits], metrics = [dice_coef, 'accuracy'])


#%%
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












