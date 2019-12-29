import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn import tree as treesklearn
from random import shuffle
from DTNode import DTNode
from MyDecisionTreeClassifier import MyDecisionTreeClassifier
from graphviz import Digraph
import os 
os.environ["PATH"] += os.pathsep + "C://Program Files (x86)//Graphviz2.38//bin"


class DesicionTree:
    def __init__(self, filename):
        self.filename = filename

    def myfunc(self):
        print("Hello my name is ")

    def load_file_cleveland(self):
        numeric_features = ['age', 'trestbps', 'chol', "thalach", "oldpeak"]
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal','num']
        col=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
        data = pd.read_csv('cleveland.data',header=None,names=col)
        return data

    def clear(self,datainclevland):
        datainclevland = datainclevland[(datainclevland['num']==0)|(datainclevland['num']==1)]
        datainclevland = datainclevland[(datainclevland['ca']!='?')&(datainclevland['thal']!='?')]
        datainclevland['ca'] = pd.to_numeric(datainclevland['ca'])
        datainclevland['thal'] = pd.to_numeric(datainclevland['thal'])
        return datainclevland

    def cleanavg(self,data):
        data = data[data['num'] < 2]
        data.fillna(data.mean(), inplace=True)
        return data

    def showChart(self,data):
        sns.distplot(data)
        plt.show()
    
    def featurearray(self):
        array=[]
        array.append({"name":"age","value":1})
        array.append({"name":"sex","value":2})
        array.append({"name":"cp","value":3})
        array.append({"name":"trestbps","value":4})
        array.append({"name":"chol","value":5})
        array.append({"name":"fbs","value":6})
        array.append({"name":"restecg","value":7})
        array.append({"name":"exang","value":8})
        array.append({"name":"oldpeak","value":9})
        array.append({"name":"slope","value":10})
        array.append({"name":"ca","value":11})
        array.append({"name":"thal","value":12})
        newarray=pd.DataFrame(array)
        return newarray
    
    
    def sklearngini(self,data):
        # Fitting and plotting by scikit-learn
        x = data.iloc[:, :-1].values
        clf = treesklearn.DecisionTreeClassifier(criterion="gini")
        clf = clf.fit(x, data.num)
        treesklearn.plot_tree(clf)
        plt.show()

    def sklearninformationgain(self,data):
        # Fitting and plotting by scikit-learn
        x = data.iloc[:, :-1].values
        clf = treesklearn.DecisionTreeClassifier(criterion="entropy")
        clf = clf.fit(x, data.num)
        treesklearn.plot_tree(clf)
        plt.show()


    def calc_misclassification_rate(self,training_dataframe, validation_dataframe):
        err = 0
        #make a object of class for use them
        dt = MyDecisionTreeClassifier()
        #list of data
        x = training_dataframe[['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']]
        #list of answer
        y = training_dataframe['num']
        dt.fit(x, y)
        b = Digraph('T', filename='Tohid')
        dt.plot(b)
        b.view()
        
        for i in validation_dataframe.index:
            if(dt.root.evaluate(validation_dataframe.loc[i, validation_dataframe.columns != "num"]) != validation_dataframe.loc[i, "num"]):
                err +=1
        #mohasebe error rate
        err = err/len(validation_dataframe)
        #show error rate
        print(err)
        return err

    def cross_validate(self,dataframe, k=5):
        #khandane data az cleveland.data
        tree=DesicionTree("cleveland.data")
        #meghdare vorudi az datafram ro index mikonam
        indices_list = list(dataframe.index)
        #shuffling data to randomize their order
        shuffle(indices_list)
        #len of shuffle data
        ill = len(indices_list)
        #len of test list
        fl = int(ill/k)
        #divided data to test data to cross validation
        validation_indices = [indices_list[i*fl:(i+1)*fl] for i in range(0, k)]
        #be ezaie har test data meghdare error rate ra hesab mikonim
        for i in range(0, k):
            #get trainin list without test
            training_indices = [x for x in indices_list if x not in validation_indices[i]]
            #calculate tree error
            tree.calc_misclassification_rate(dataframe.loc[training_indices, :], dataframe.loc[validation_indices[i], :])
    
        
