# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:31:41 2019

@author: Bahar
"""



import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import train_test_split


N= 64
num_classes=22
batch_size=50

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(N,N,3), padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3, 3), strides=2))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(128, (2, 2), strides=2))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())


model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)


test_gen = ImageDataGenerator()

n_folds=10
model_history = [] 

for i in range(n_folds):
    print("Training on Fold: ",i+1)
    t_x, val_x, t_y, val_y = train_test_split(train_X, train_y, test_size=0.1, random_state = np.random.randint(1,1000, 1)[0])
    
    train_generator = gen.flow(t_x, t_y, batch_size=batch_size)
    test_generator = test_gen.flow(val_x, val_y, batch_size=batch_size)
    
    model_t= model.fit_generator(train_generator, steps_per_epoch=t_x.shape[0] // batch_size, epochs=10, 
                    validation_data=test_generator, validation_steps=val_x.shape[0] // batch_size, 
                    callbacks=[learning_rate_reduction])
    model_history.append(model_t)
    final_loss, final_acc = model.evaluate(val_x, val_y, verbose=0)
    print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
    print("======="*12, end="\n\n\n")
#==============================================================================
#print(model_t.history.keys())
accuracy=[]
val_accuracy=[]
loss=[]
val_loss=[]
epochs=[]
col=['b','g','r','c','m','y','k','w']
for i in range(n_folds):
    accuracy.append(model_history[i].history['acc'])
    val_accuracy.append(model_history[i].history['val_acc'])
    loss.append(model_history[i].history['loss'])
    val_loss.append(model_history[i].history['val_loss'])
    epochs.append(range(len(accuracy)))

    plt.plot(epochs[i], accuracy[i], 'bo', label='Train accuracy Fold '+str(i+1), color=col[i])
    plt.plot(epochs[i], val_accuracy[i], 'b', label='Val accuracy Fold '+str(i+1), color=col[i])
    
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
plt.figure()

for i in range(n_folds):
    plt.plot(epochs[i], loss[i], 'bo', label='Train loss Fold '+str(i+1), color=col[i] )
    plt.plot(epochs[i], val_loss[i], 'b', label='Val loss Fold '+str(i+1), color=col[i])
    
plt.title('Training and validation loss')
plt.legend()
plt.show()

plt.title('Accuracies vs Epoch')
for i in range(n_folds):
    plt.plot(model_history[0].history['acc'], label='Training Fold '+str(i+1))

plt.legend()
plt.show()
#==============================================================================
#Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#------------------------------------------------------------------------------
Y_pred_test = model.predict_classes(test_X)
Y_pred_classes_test = np.argmax(Y_pred_test, axis = 1) 
Y_true_test = np.argmax(test_y, axis = 1) 
confusion_mtx_test = confusion_matrix(Y_true_test, Y_pred_classes_test) 
plot_confusion_matrix(confusion_mtx_test, classes = range(num_classes))
plt.show()
correct = np.nonzero(Y_pred_test==test_y)[0]
incorrect = np.nonzero(Y_pred_test!=test_y)[0]
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_y, Y_pred_test, target_names=target_names))
#==============================================================================
errors = (Y_pred_classes_test - Y_true_test != 0)

Y_pred_classes_errors = Y_pred_classes_test[errors]
Y_pred_errors = Y_pred_test[errors]
Y_true_errors = Y_true_test[errors]
X_test_errors = test_X[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((N,N)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_dela_errors[-6:]

display_errors(most_important_errors, X_test_errors, Y_pred_classes_errors, Y_true_errors)
print("Probabilities of the wrong predicted numbers: ", Y_pred_errors_prob)
print("Predicted probabilities of the true values in the error set: ", true_prob_errors)
