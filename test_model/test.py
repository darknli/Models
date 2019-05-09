from __future__ import print_function
# 导入numpy库， numpy是一个常用的科学计算库，优化矩阵的运算
import numpy as np
np.random.seed(1337)


# 导入mnist数据库， mnist是常用的手写数字库
# 导入顺序模型
from tensorflow.python.keras.models import Sequential
# 导入全连接层Dense， 激活层Activation 以及 Dropout层
# from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
# 设置batch的大小
batch_size = 100
# 设置类别的个数
nb_classes = 10
# 设置迭代的次数
nb_epoch = 20

'''
下面这一段是加载mnist数据，网上用keras加载mnist数据都是用
(X_train, y_train), (X_test, y_test) = mnist.load_data()
但是我用这条语句老是出错：OSError: [Errno 22] Invalid argument
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)/255
x_test = x_test.reshape(10000, 784)/255
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)

def get_model():
    inputs = Input((784, ))
    net = Dense(500, activation="relu")(inputs)
    net = Dense(500, activation="relu")(net)
    print(net.shape[-1])
    # net = Dense(10, activation="softmax")(net)
    return Model(inputs=inputs, outputs=net)

from keras.layers import TimeDistributed


base = get_model()
inp = Input((784,))
net = base(inp)
net = Dense(10, activation="softmax")(net)
model = Model(inputs=inp, outputs=net)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
history = model.fit(x_train, y_train,
                    batch_size = 200,
                    epochs = 20,
                    verbose = 1,
                    validation_data = (x_test, y_test))

score = model.evaluate(x_test, x_test, verbose=0)

# 输出训练好的模型在测试集上的表现
print('Test score:', score[0])
print('Test accuracy:', score[1])