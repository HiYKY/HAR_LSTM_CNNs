#=========================== import tool packages ====================================================
import numpy as np
np.random.seed(42)

import tensorflow as tf
#tf.set_random_seed(42) #uses in old version, use tf.compat.v1 to replace tf. Or use tf.random.set_seed(42) in new version.
tf.compat.v1.set_random_seed(42)

import matplotlib.pyplot as plt

# for reproducibility
# https://github.com/fchollet/keras/issues/2280
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

import keras
#from keras import backend as K
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, GRU, TimeDistributed
from keras.layers.core import Dense, Dropout
#=========================== import tool packages END ====================================================


#=========================== define functions ====================================================
def _count_classes(y):
    return len(set([tuple(category) for category in y]))
#=========================== define functions END ====================================================


#=========================== parameters setting ====================================================
#---------- train and test data ----------------
from data import load_data

X_train, X_test, Y_train, Y_test = load_data()
#---------- train and test data END ----------------

#--------- parameters for model ----------
timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)

epochs = 30
batch_size = 16
n_hidden = 32
#--------- parameters for model END----------
#=========================== parameters setting END ====================================================


#=========================== develop,train and test LSTM model ====================================================

#--------------- 定义模型 create model ---------------
model = Sequential()
model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
model.add(Dense(n_classes, activation='sigmoid'))
model.summary()


'''
##参数
#n_hidden=32 #隐层节点数
#timesteps=128 #lstm第一层有128个输入神经元节点
#n_classes=6 #分类的目标，有6类
'''

'''
keras.layers.core.Dense(
units, #代表该层的输出维度
activation=None, #激活函数.但是默认 liner
use_bias=True, #是否使用b
kernel_initializer='glorot_uniform', #初始化w权重，keras/initializers.py
bias_initializer='zeros', #初始化b权重
kernel_regularizer=None, #施加在权重w上的正则项,keras/regularizer.py
bias_regularizer=None, #施加在偏置向量b上的正则项
activity_regularizer=None, #施加在输出上的正则项
kernel_constraint=None, #施加在权重w上的约束项
bias_constraint=None #施加在偏置b上的约束项
)
'''
#--------------- 定义模型 create model END ---------------

#--------------- 编译模型 Compile model ---------------
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#--------------- 编译模型 Compile model END ---------------

#--------------- 训练模型 ---------------
model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          validation_data=(X_test, Y_test),
          epochs=epochs)
#--------------- 训练模型 END ---------------

#--------------- 评估模型 ---------------
train_loss, train_accuracy = model.evaluate(X_train,Y_train) 
print("train_loss =",train_loss)
print("train_accuracy =",train_accuracy)

test_loss, test_accuracy = model.evaluate(X_test,Y_test) 
print("test_loss =",test_loss)
print("test_accuracy =",test_accuracy)

# confusion_matrix
from utils import confusion_matrix

Confusion_Matrix = confusion_matrix(Y_test, model.predict(X_test))
print(Confusion_Matrix)
Confusion_Matrix.to_csv("confusionmatrix.csv")
#--------------- 评估模型 END ---------------

#=========================== develop,train and test LSTM model END ====================================================


#=========================== Plot Results ====================================================
# Output classes to learn how to classify
LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
] 

width = 10
height = 10
plt.figure(figsize=(width, height))
plt.imshow(
    Confusion_Matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)

plt.title("HAR Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('1.png', dpi=480)#保存图片
plt.show()
#=========================== Plot Results END ====================================================

