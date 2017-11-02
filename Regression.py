import numpy as np
import matplotlib.pyplot as plt

#定义存储输入数据（x）和目标数据（y）的数组
x, y = [], []
for sample in open("E:\price.txt","r"):
    _x, _y =sample.split(",") #用“,”分开数据集
    #将字符串数据转化为浮点数，并放入数组中
    x.append(float(_x))
    y.append(float(_y))
x, y =np.array(x), np.array(y) #将数组转化为Numpy数组
#标准化，将大数字简化为小数字，方便处理。（x-x的平均值）/x的标准差
x = (x-x.mean()) / x.std()
#画出散点图
plt.figure()
plt.scatter(x,y,c="g",s=6)
plt.show()    #以上完成数据预处理

x0 = np.linspace(-2, 4, 100) #在(-2,4)区间内取100个点
def get_model(deg):
	return lambda input_x=x0: np.polyval(np.polyfit(x, y, deg), input_x)

def get_cost(deg, input_x, input_y):
	return 0.5*((get_model(deg)(input_x) - input_y)**2).sum()

test_set = (1, 4, 10)
for d in test_set:
	print(get_cost(d, x ,y))

plt.scatter(x, y, c="g", s=20)
for d in test_set:
	plt.plot(x0, get_model(d)(),label="degree = {}".format(d))

plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)

plt.legend()
plt.show()