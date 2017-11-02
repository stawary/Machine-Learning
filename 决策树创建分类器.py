from sklearn import tree  #决策树
features=[[140,1],[130,1],[150,0],[170,0]]
labels=[0,0,1,1]
clf=tree.DecisionTreeClassifier() #用决策树初始化一个分类器
clf=clf.fit(features,labels) #用training data去训练分类器
print clf.predict([[120,0]]) #给出test data,得到新的labels