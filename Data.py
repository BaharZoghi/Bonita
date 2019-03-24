# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:16:31 2019

@author: Bahar
"""

import glob
from skimage import io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import Markdown, display
import pandas as pd
import sys, os
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


img_size = 64

path_X = 'C:/Data/Hack/Data/crop_part1/crop_part1/*.jpg'

protected_race = []
outcome_gender = []
feature_image = []
feature_age = []

files = glob.glob(path_X)
for i, image_path in enumerate(files):
    age, gender, race = image_path.split('\\')[-1].split("_")[:3]
    outcome_gender.append(gender)
    protected_race.append(race)
    feature_image.append(resize(io.imread(image_path), (img_size, img_size)))
    feature_age.append(int(age))
    
feature_image_mat = np.array(feature_image)
outcome_gender_mat =  np.array(outcome_gender)
protected_race_mat =  np.array(protected_race)
age_mat = np.array( list(map(int, feature_age)))

feature_image_mat_normed = 2.0 *feature_image_mat.astype('float32') - 1.0

age_train=[]

N = len(feature_image_mat_normed)
ids = np.random.permutation(N)
train_size=int(0.7 * N)

X_train = feature_image_mat_normed[ids[0:train_size]]
y_train = outcome_gender_mat[ids[0:train_size]]

X_test = feature_image_mat_normed[ids[train_size:]]
y_test = outcome_gender_mat[ids[train_size:]]

p_train = protected_race_mat[ids[0:train_size]]
p_test = protected_race_mat[ids[train_size:]]

age_train= age_mat[ids[0:train_size]] 
age_test = age_mat[ids[train_size:]]


for i in range(len(age_train)):
    if (age_train[i]>=0 and age_train[i]<5):
        age_train[i]=2
    elif (age_train[i]>=5 and age_train[i]<10):
        age_train[i]=7
    elif (age_train[i]>=10 and age_train[i]<15):
        age_train[i]=12
    elif (age_train[i]>=15 and age_train[i]<20):
        age_train[i]=17
    elif (age_train[i]>=20 and age_train[i]<25):
        age_train[i]=22
    elif (age_train[i]>=25 and age_train[i]<30):
        age_train[i]=27
    elif (age_train[i]>=30 and age_train[i]<35):
        age_train[i]=32
    elif (age_train[i]>=35 and age_train[i]<40):
        age_train[i]=37
    elif (age_train[i]>=40 and age_train[i]<45):
        age_train[i]=42
    elif (age_train[i]>=45 and age_train[i]<50):
        age_train[i]=47
    elif (age_train[i]>=50 and age_train[i]<55):
        age_train[i]=52
    elif (age_train[i]>=55 and age_train[i]<60):
        age_train[i]=57
    elif (age_train[i]>=60 and age_train[i]<65):
        age_train[i]=62
    elif (age_train[i]>=65 and age_train[i]<70):
        age_train[i]=67
    elif (age_train[i]>=70 and age_train[i]<75):
        age_train[i]=72
    elif (age_train[i]>=75 and age_train[i]<80):
        age_train[i]=77
    elif (age_train[i]>=80 and age_train[i]<85):
        age_train[i]=82
    elif (age_train[i]>=85 and age_train[i]<90):
        age_train[i]=87
    elif (age_train[i]>=90 and age_train[i]<95):
        age_train[i]=92
    elif (age_train[i]>=95 and age_train[i]<100):
        age_train[i]=97
    elif (age_train[i]>=100 and age_train[i]<105):
        age_train[i]=102
    elif (age_train[i]>=105 and age_train[i]<110):
        age_train[i]=107
    else:
        age_train[i]=113
        

for i in range(len(age_test)):
    if (age_test[i]>=0 and age_test[i]<5):
        age_test[i]=2
    elif (age_test[i]>=5 and age_test[i]<10):
        age_test[i]=7
    elif (age_test[i]>=10 and age_test[i]<15):
        age_test[i]=12
    elif (age_test[i]>=15 and age_test[i]<20):
        age_test[i]=17
    elif (age_test[i]>=20 and age_test[i]<25):
        age_test[i]=22
    elif (age_test[i]>=25 and age_test[i]<30):
        age_test[i]=27
    elif (age_test[i]>=30 and age_test[i]<35):
        age_test[i]=32
    elif (age_test[i]>=35 and age_test[i]<40):
        age_test[i]=37
    elif (age_test[i]>=40 and age_test[i]<45):
        age_test[i]=42
    elif (age_test[i]>=45 and age_test[i]<50):
        age_test[i]=47
    elif (age_test[i]>=50 and age_test[i]<55):
        age_test[i]=52
    elif (age_test[i]>=55 and age_test[i]<60):
        age_test[i]=57
    elif (age_test[i]>=60 and age_test[i]<65):
        age_test[i]=62
    elif (age_test[i]>=65 and age_test[i]<70):
        age_test[i]=67
    elif (age_test[i]>=70 and age_test[i]<75):
        age_test[i]=72
    elif (age_test[i]>=75 and age_test[i]<80):
        age_test[i]=77
    elif (age_test[i]>=80 and age_test[i]<85):
        age_test[i]=82
    elif (age_test[i]>=85 and age_test[i]<90):
        age_test[i]=87
    elif (age_test[i]>=90 and age_test[i]<95):
        age_test[i]=92
    elif (age_test[i]>=95 and age_test[i]<100):
        age_test[i]=97
    elif (age_test[i]>=100 and age_test[i]<105):
        age_test[i]=102
    elif (age_test[i]>=105 and age_test[i]<110):
        age_test[i]=107
    else:
        age_test[i]=113
        
N = 64
train_X = X_train.reshape(X_train.shape[0], N, N, 3)
test_X = X_test.reshape(X_test.shape[0], N, N, 3)


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')


train_X/=255.0
test_X/=255.0


label_encoder = LabelEncoder()
test_y = label_encoder.fit_transform(age_test)
train_y = label_encoder.fit_transform(age_train)


onehot_encoder = OneHotEncoder(sparse=False)
train_y = train_y.reshape(len(train_y), 1)
train_y = onehot_encoder.fit_transform(train_y)        
      
        
test_y= np.array(test_y)
train_y= np.array(train_y)


train_y = train_y.astype('float32')
test_y = test_y.astype('float32')
