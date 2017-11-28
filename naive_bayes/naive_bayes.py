#本系列均以sklearn书写各算法的应用 

from sklearn.naive_bayes import GaussianNB #http://scikit-learn.org/stable/modules/naive_bayes.html
from sklearn.metrics import accuracy_score #http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
                                           #计算结果的准确率
from time import time #用以查看训练和预测所用的时间

features_train,labels_train,features_test, labels_test =  #数据源自己定义

clf = GaussianNB() #生成分类器，参数可查看文档
t0 = time()
clf.fit(features_train,labels_train) #训练
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test) #预测
print ("predicting time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(labels_test, pred) #得到准确率
print(accuracy)


