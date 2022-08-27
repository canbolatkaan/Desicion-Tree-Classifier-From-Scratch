class Node:
    def __init__(self, featureIndex = None, threshold=None,left=None, right=None,info_gain=None, value=None):
        self.featureIndex = featureIndex
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value
        
from typing import List

class DecisionTreeClassifier:
    def __init__(self, max_depth: int):
        
        self.max_depth= max_depth
        self.rootNode = None
        

    def fit(self, X: List[List[float]], y: List[int]):
        dataSet= self.concataneLists(X,y)
        self.rootNode= self.decisionTree(dataSet,0)
        

    def predict(self, X: List[List[float]]):
        preditions = [self.make_prediction(x, self.rootNode) for x in X]
        return preditions
    
    def concataneLists(self,list1: List[List[float]],list2 : List[int]):
        count=0
        for row in list1:
            row.append(list2[count])
            count=count+1
            
        return list1
    def deConcatanateY(self,dataSet):
        y=[];
        
        count=0
        for row in dataSet:
            y.append(int(row[len(row)-1]));
            
        
        return y
    def get_unique_list(self,myList):
        unique_list=[]
        for x in myList:
            if x not in unique_list:
                unique_list.append(x)
        return unique_list
    def decisionTree(self,dataSet,curr_depth):
        y=self.deConcatanateY(dataSet)
        
        
        samples= len(dataSet)
        features = len(dataSet[0])-1
        
        if(samples >= 3 and  curr_depth <= self.max_depth):
            
            bSplit = self.get_best_split(dataSet,features)
            
            if(bSplit["informainGain"]>0):
                left_tree = self.decisionTree(bSplit["left"], curr_depth+1)
                right_tree= self.decisionTree(bSplit["right"], curr_depth+1)
                return Node(bSplit["column"], bSplit["threshold"], left_tree,right_tree,bSplit["informainGain"])
        leafVal= max(y, key=y.count)
        
        return Node(value=leafVal)
    def get_best_split(self,dataset,features):
        best_split ={}
        maxGain=0
        best_split["informainGain"] =0
        current_uncertainity=self.gini(dataset)
        for index in range(features):
            features_column= self.get_Column(dataset,index)
            thresholds= sorted(self.get_unique_list(features_column))
            for threshold in thresholds:
                dataset_left,dataset_right = self.splitWithThreshold(dataset,index,threshold) 
                
                if len(dataset_left)>0 and len(dataset_right)>0 :
                    
                    informationGain=self.info_gain(dataset_left,dataset_right,current_uncertainity)

                    if informationGain > maxGain:
                        best_split["column"] = index
                        best_split["threshold"] = threshold
                        best_split["left"] = dataset_left
                        best_split["right"] = dataset_right
                        best_split["informainGain"] = informationGain
                        maxGain = informationGain
        return best_split
    
    def splitWithThreshold(self,dataset,index,threshold):
        data_temp_left = []
        data_temp_right = []
        data_left = []
        data_right = []
        splitted_column_values = []
        for k in range (len(dataset)):
            splitted_column_values.append(dataset[k][index])
        
        for k in range (len(splitted_column_values)):
            if splitted_column_values[k] <= threshold:
                for i in range (len(dataset[0])):
                    data_temp_left.append(dataset[k][i])
                    
                data_left.append(data_temp_left)   
                data_temp_left = []
                
            else: 
                 for i in range (len(dataset[0])):
                    data_temp_right.append(dataset[k][i])
                    
                 
                 data_right.append(data_temp_right)
                 data_temp_right = []
        
        return data_left,data_right 
    def get_Column(self,List,index):
        

        myList=[]
        for row in List:
            myList.append(float(row[index]))
        return myList
    
    def class_counts(rows):
        counts = {} 
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts
    
    #Gini impurity
    def gini(self,rows):
  
        counts = {}  
        for row in rows:
            
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity
    
    #Gini impurity'e göre information gain hesabı
    def info_gain(self,left, right, current_uncertainty):

        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: 
            return tree.value
        feature_val = x[tree.featureIndex]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
            

                     
# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)    
    