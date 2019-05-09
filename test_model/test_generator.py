import os
import cv2


import random
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

def ch_fn(mat):
    return mat/128.0 - 1
train_datagen = ImageDataGenerator(
    # rescale=1.0/255
preprocessing_function=ch_fn
)

train_generator = train_datagen.flow_from_directory(
    os.path.join('D:\\AAA\\work', 'xuesongpic'),
    classes=['1','2'],
    batch_size=1,
    class_mode='binary')

for i, (x, y) in enumerate(train_generator):
    x = np.squeeze(x, axis=0)
    y = y.reshape(-1)[0]
    print(np.max(x), np.min(x), np.mean(x))
    # print(x.shape, y)
    # cv2.imwrite(str(random.random())+'_'+str(y)+'.jpg', x)
    # if i > 5 :
    #     break