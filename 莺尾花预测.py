import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0,50,100] #从150组数据中提取每种莺尾花的第一组数据

#training data
train_target = np.delete(iris.target,test_idx) #去掉第0,50,100组数据
train_data = np.delete(iris.data,test_idx,axis=0)

#testing data
test_target = iris.target[test_idx] #用去掉的莺尾花数据作为测试数据 
test_data = iris.data[test_idx]

clf=tree.DecisionTreeClassifier() #用决策树初始化一个分类器
clf=clf.fit(train_data,train_target)

print (test_target )#实际结果
print (clf.predict(test_data))#训练后的预测结果

#viz code 可视化代码
import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf") #当前目录下生成pdf
