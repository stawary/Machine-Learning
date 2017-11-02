#基于多层感知器的softmax多分类问题
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data 生成虚拟数据
import numpy as np
x_train = np.random.random((1000, 20)) #随机一个1000×20的数组。1000个样本，每个样本有20个特征值
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10) #输出10种lable（种类）
x_test = np.random.random((100, 20))            #np.random.randint(10, size=(1000, 1)) 随机生成一个1000×1的数组，取值范围[0,10)
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential() #序贯模型，是多个网络层的线性堆叠，也就是“一条路走到黑”。
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))#第一个隐藏层为64个神经元的全连接层，输入20个特征值（输入层），激活函数relu
model.add(Dropout(0.5)) #Dropout 以概率 p 来丢弃神经元， 并且让别的神经元以概率 q = 1 - p，进行保留。
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))#用softmax对上一层64个神经元进行激活，得到10个输出lable（输出层）

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #机梯度下降SGD。lr=learning rate ,decay是学习速率的衰减系数
                                  #  momentum表示动量项。  Nesterov的值是False或者True，表示使不使用Nesterov momentum。 
model.compile(loss='categorical_crossentropy',    #损失函数 多类分类的交叉熵
              optimizer=sgd,   # 优化器 随机梯度下降，其他如'rmsprop'、'adagrad'
              metrics=['accuracy'])  #指标列表metrics  对分类问题，我们一般将该列表设置为metrics=['accuracy']

model.fit(x_train, y_train,
          epochs=20,   #迭代次数20次，一个epoch是指把所有训练数据完整的过一遍。
          batch_size=128,#指的是每一次迭代的训练，使用数据的个数
          shuffle=True, #shuffle就是是否把数据随机打乱之后再进行训练
          validation_split=0.3#validation_split就是拿出百分之多少用来做交叉验证
          )  
print("test set")
score = model.evaluate(x_test, y_test, batch_size=128,verbose=1) #,verbose=0、1、2。 0是不屏显，1是显示一个进度条，2是每个epoch都显示一行数据

#以下代码计算正确率
result = model.predict(x_test,batch_size=200,verbose=1)

result_max = np.argmax(result, axis = 1)
test_max = np.argmax(y_test, axis = 1)

result_bool = np.equal(result_max, test_max)
true_num = np.sum(result_bool)
print("")
print("The accuracy of the model is %f" % (true_num/len(result_bool)))

#此模型输入输出均随机，网络层也不够深，故正确率不高。用来理解keras用法                    