
from DTNode import DTNode
from collections import deque

class MyDecisionTreeClassifier:
    
    def __init__(self, criterion="entropy"):
        #criterion = entropy | gini
        self.criterion = criterion
        self.root = None
        
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.root = DTNode(x, y, criterion=self.criterion)

    def plot(self, graph):
        # estefade az safe 2 tarage baraie peimaiesh derakht
        q = deque()
        #afzudan be safe movaghat
        q.append(self.root)
        while(q):
            # bardashtan meghdare aval
            node = q.pop()
            #agar barg bud
            if(node.isleaf):
                graph.node(str(id(node)), str(node.target_class) + str(node.check))
            else:
                graph.node(str(id(node)), str(node.feature) + "\n" + str(node.devision_thresholds) + str(node.check))
            for child in node.children:
                graph.edge(str(id(node)), str(id(child)))
                q.append(child) 

    def prone(self, graph):
        # estefade az safe 2 tarage baraie peimaiesh derakht
        q = deque()
        #all nodes
        allnodes =[]
        #afzudan be safe movaghat
        q.append(self.root)
        #afzudan be safe daeem
        allnodes.append(self.root)

        while(q):
            # bardashtan meghdare aval
            node = q.pop()
            #agar barg bud
            if(node.isleaf):
                graph.node(str(id(node)), str(node.target_class) + str(node.check))
            else:
                #graph node
                graph.node(str(id(node)), str(node.feature) + "\n" + str(node.devision_thresholds) + str(node.check))
            for child in node.children:
                #graph node
                graph.edge(str(id(node)), str(id(child)))
                #afzudane farzandan be safe movaghat
                q.append(child) 
                #afzudan be safe daeem
                allnodes.append(child)
        return allnodes

