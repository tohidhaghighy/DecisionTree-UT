import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from AllClasses import DesicionTree
from sklearn import tree as treesklearn
from DTNode import DTNode
from MyDecisionTreeClassifier import MyDecisionTreeClassifier
from graphviz import Digraph
import os 
os.environ["PATH"] += os.pathsep + "C://Program Files (x86)//Graphviz2.38//bin"


tree=DesicionTree("cleveland.data")
datainclevland=tree.load_file_cleveland()

data=tree.clear(datainclevland)
x = data.iloc[:, :-1].values
y = data.num

# print(x)
# print(y)
categorical_features = data[['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']]

# newnode=MyDecisionTreeClassifier()
# newnode.fit(categorical_features, data.num)
# b = Digraph('T', filename='Tohid')
# newnode.plot(b)
# b.view()

#tree.cross_validate(data,5)

#root = DTNode(categorical_features, data.num)
# print(root)
# #print(tree.Id3(datainclevland,tree.featurearray()))
#data=tree.clear(datainclevland)
# print(data)
tree.sklearngini(data)
tree.sklearninformationgain(data)


# # print(datainclevland)
# #print(tree.Id3(datainclevland,tree.featurearray()))





# # g = Digraph('G', filename='hello.gv')
# # g.edge('Hello', 'World')
# # g.view()








# print("رسم نمودار age")
# tree.showChart(datainclevland.age)

# print("رسم نمودار sex")
# tree.showChart(datainclevland.sex)

# print("رسم نمودار cp")
# tree.showChart(datainclevland.cp)

# print("رسم نمودار trestbps")
# tree.showChart(datainclevland.trestbps)

# print("رسم نمودار chol")
# tree.showChart(datainclevland.chol)

# print("رسم نمودار fbs")
# tree.showChart(datainclevland.fbs)

# print("رسم نمودار restecg")
# tree.showChart(datainclevland.restecg)

# print("رسم نمودار thalach")
# tree.showChart(datainclevland.thalach)

# print("رسم نمودار exang")
# tree.showChart(datainclevland.exang)

# print("رسم نمودار oldpeak")
# tree.showChart(datainclevland.oldpeak)

# print("رسم نمودار slope")
# tree.showChart(datainclevland.slope)

# print("رسم نمودار ca")
# #tree.showChart(datainclevland.ca)

# print("رسم نمودار num")
# tree.showChart(datainclevland.num)

# print("رسم نمودار thal")
# tree.showChart(datainclevland.thal)


