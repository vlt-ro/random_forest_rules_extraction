import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import ceil
from matplotlib.patches import Rectangle
from sklearn.utils import  check_random_state
from sklearn.metrics import confusion_matrix


class Node:
    def __init__(self,depth,gini=None,attribute_index = None,th=None,left_tree=None,right_tree=None,leaf_value=None):

        self.gini_index = gini
        self.th = th
        self.attribute_index = attribute_index
        self.left_tree = left_tree
        self.right_tree = right_tree
        self.leaf_value = leaf_value
        self.depth = depth

class DecisionTree:
    def __init__(self,max_depth=10,min_group_size=10,percen_M = 1,replacement = None, random_state=42):
        self.replacement = replacement
        self.percen_M = percen_M
        self.max_depth = max_depth
        self.min_group_size = min_group_size
        self.random_state = random_state
        self.df_tree = []


    def fit(self,X,y,feat_info=None,feat_name=None):

        if feat_info == None:
            self.feat_info = ['continuous']*X.shape[1]
        else:
            self.feat_info = feat_info

        if feat_name == None:
            self.feat_name = ['X'+str(i) for i in range(X.shape[1])]
        else:
            self.feat_name = feat_name

        self.root = self.growth_tree(X,y,depth=1)

    def plot_tree(self,X,y,save=False,ds='dt'):
        color = ['C0','C1']
        tree = self.root
        self.df_tree = [[0,X[:, 0].min()],
                        [1,X[:, 1].min()],
                        [0,X[:, 0].max()],
                        [1,X[:, 1].max()]]
        self.get_info_tree(tree,True)
        self.df_tree = pd.DataFrame(self.df_tree,columns=['feat_index','th'])

        plt.figure(figsize=(10,7))
        self.ax = plt.subplot(111)

        arr_x = self.df_tree[self.df_tree.feat_index==0].sort_values(by='th').th.values
        arr_y = self.df_tree[self.df_tree.feat_index==1].sort_values(by='th').th.values
        for ix in range(len(arr_x)-1):
            for iy in range(len(arr_y)-1):

                pred = self.predict(np.array([[(arr_x[ix]+arr_x[ix+1])/2,(arr_y[iy]+arr_y[iy+1])/2]]))[0]
                self.ax.add_patch(Rectangle((arr_x[ix], arr_y[iy]),
                                        arr_x[ix+1]-arr_x[ix], arr_y[iy+1]-arr_y[iy],
                                        facecolor = color[pred],alpha=0.3))

        sns.scatterplot(x = X[:, 0],y= X[:, 1], hue=y)

        self.ax.legend()
        self.ax.set_xlabel(self.feat_name[0])
        self.ax.set_ylabel(self.feat_name[1])

        if save:
            plt.tight_layout()
            plt.savefig('results/decisiontree'+ds+'.png')
        else :
           plt.show()

    def get_info_tree(self,tree,graph = False,parent_index=None,parent_th=None,side=None):

        if tree.leaf_value != None:
            if not(graph):
                self.df_tree.append([None,None,parent_index,parent_th,side,tree.leaf_value])

            pass
        else:
            if graph:
                self.df_tree.append([tree.attribute_index,tree.th])
            else:
                self.df_tree.append([tree.attribute_index,tree.th,parent_index,parent_th,side,None])

            self.get_info_tree(tree.left_tree,graph,tree.attribute_index,tree.th,'left')
            self.get_info_tree(tree.right_tree,graph,tree.attribute_index,tree.th,'right')

    def predict(self,X):
        y_pred = []
        for X_v in X:
            y_pred.append(self.make_prediction(X_v,self.root))
        return y_pred

    def make_prediction(self,X,tree):

        if tree.leaf_value!=None:
            return tree.leaf_value

        idx = tree.attribute_index
        th = tree.th
        if X[idx]<th:
            return self.make_prediction(X, tree.left_tree)
        else:
            return self.make_prediction(X, tree.right_tree)

    def get_indices(self,n,percen):
        random_instance = check_random_state(self.random_state)
        index = random_instance.randint(0, n, int(percen*n))
        sample_counts = np.bincount(index, minlength=n)
        unsampled_mask = sample_counts == 0
        indices_range = np.arange(n)
        unsampled_indices = indices_range[unsampled_mask]

        return unsampled_indices

    def growth_tree(self,X,y,depth):

        if depth <= self.max_depth and len(X)>=self.min_group_size:
            best_node_split = self.split_node(X,y)

            n_left = len(best_node_split['left_data']['X'])
            n_right = len(best_node_split['right_data']['X'])

            if self.replacement!=None and self.percen_M<1:
                #index_left = self.get_indices(n_left,self.percen_M)
                #index_right = self.get_indices(n_right,self.percen_M)

                index_left = np.random.choice(n_left, int(self.percen_M*n_left), replace=self.replacement)
                index_right = np.random.choice(n_right, int(self.percen_M*n_right), replace=self.replacement)
            else:
                index_left = np.arange(n_left)
                index_right = np.arange(n_right)

            X_left = best_node_split['left_data']['X'][index_left]
            y_left = best_node_split['left_data']['y'][index_left]
            X_right = best_node_split['right_data']['X'][index_right]
            y_right = best_node_split['right_data']['y'][index_right]

            if len(X_left)<=self.min_group_size or len(X_right)<=self.min_group_size:# or gini>gini_prev:
                return Node(depth,leaf_value=self.leaf_value(y))
            else:
                left_tree = self.growth_tree(X_left,y_left,depth+1)
                right_tree = self.growth_tree(X_right,y_right,depth+1)

                return Node(depth,best_node_split['gini'],best_node_split['index_attribute'],best_node_split['th'],
                            left_tree=left_tree, right_tree=right_tree)

        return Node(depth,leaf_value=self.leaf_value(y))

    def leaf_value(self,y):

        values, values_counts = np.unique(y,return_counts=True)
        return values[values_counts==max(values_counts)][0]

    def split_node(self,X,y):
        mini_gini = 999
        for index_attribute in range(X.shape[1]):

            X_n = X[:,index_attribute]

            for th in np.unique(X_n):

                y_g1, y_g2 = self.split_data(X,y,index_attribute,th)

                gini = self.gini_impurity(y_g1,y_g2)

                if gini<mini_gini:
                    mini_gini = gini
                    best_th = th
                    best_attribute_index = index_attribute

        X_g1, y_g1,X_g2,y_g2= self.split_data(X,y,best_attribute_index,best_th,True)
        return {'gini':self.gini_index(y),'th':best_th,'index_attribute':best_attribute_index,
                'left_data':{'X':X_g1,'y':y_g1},
                'right_data':{'X':X_g2,'y':y_g2}}

    def split_data(self, X,y,index_attribute,threshold, return_data = False):

        if self.feat_info[index_attribute] == 'continuous':
            left_idx = X[:,index_attribute]<threshold
            right_idx = X[:,index_attribute]>=threshold

        elif self.feat_info[index_attribute] == 'categorical':
            left_idx = X[:,index_attribute] == threshold
            right_idx = X[:,index_attribute] != threshold

        X_g1 = X[left_idx]
        y_g1 = y[left_idx]
        X_g2 = X[right_idx]
        y_g2 = y[right_idx]

        if return_data:
            return X_g1, y_g1, X_g2,y_g2
        else:
            return y_g1, y_g2

    def gini_impurity(self,y_g1,y_g2):
        gini1, gini2 = self.gini_index(y_g1), self.gini_index(y_g2)

        gini = (len(y_g1)*gini1+len(y_g2)*gini2)/(len(y_g1)+len(y_g2))
        return gini

    # Purity of the node
    def gini_index(self,y):
        values, values_counts = np.unique(y,return_counts=True)
        p_i = values_counts/len(y)
        return 1- np.sum(p_i**2)

class RandomForest:
    def __init__(self,percen_L = 0.7,percen_M=0.5,n_trees=10,replacement = True,max_depth=10,min_group_size=10, random_state=42):
        self.percen_L = percen_L
        self.percen_M = percen_M
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_group_size = min_group_size
        self.replacement = replacement

        self.initial_random_state = random_state
        self.random_state = check_random_state(random_state)
        self.random_state = self.random_state.randint(np.iinfo(np.int32).max, size=self.n_trees)

        self.forest = None

    def fit(self,X,y,feat_info=None,feat_name=None,extract_rules=True):
        self.forest = []

        nX = len(X)
        for n_tree in range(self.n_trees):
            tree = DecisionTree(max_depth = self.max_depth,min_group_size= self.min_group_size,
                                percen_M = self.percen_M, replacement = self.replacement,
                                random_state=self.initial_random_state)

            index = np.random.choice(nX, int(self.percen_L*nX), replace=self.replacement)
            """
            random_instance = check_random_state(self.random_state[n_tree])
            index = random_instance.randint(0, nX, int(self.percen_L*nX))
            sample_counts = np.bincount(index, minlength=nX)
            unsampled_mask = sample_counts == 0
            indices_range = np.arange(nX)
            index = indices_range[unsampled_mask]
            """
            tree.fit(X[index],y[index],feat_info,feat_name)

            if extract_rules:
                tree.get_info_tree(tree.root,False,tree.root.attribute_index,tree.root.th,'root')
                df_tree = pd.DataFrame(tree.df_tree,columns=['feat_index','th','parent_feat_index','parent_th','side','value'])
                df_tree['tree'] = n_tree

                if n_tree==0:
                    df_forest = df_tree
                else:
                    df_forest = pd.concat([df_forest,df_tree])
            self.forest.append(tree)

        return df_forest

    def predict(self,X):
        y_pred = []
        for X_v in X:
            y_pred_f = []
            for tree in self.forest:
                y_pred_f.append(tree.make_prediction(X_v,tree.root))
            y_pred.append(self.forest[0].leaf_value(y_pred_f))

        return y_pred

def error_rate(y_true, y_pred):
    confusion_matrix_ = confusion_matrix(y_true,y_pred)
    print(confusion_matrix_)
    np.fill_diagonal(confusion_matrix_,0)
    return np.round(100*np.sum(confusion_matrix_)/len(y_true),2)
