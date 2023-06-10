import numpy as np
import pandas as pd
seed = 296
np.random.seed(seed)

from sklearn.datasets import make_moons,make_circles,load_iris,fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.random_forest import DecisionTree, RandomForest, error_rate
from src.extract_rules import get_rules

datasets = {'make_moons':make_moons(n_samples=1000, noise=0.1, random_state=seed),
            'make_circles':make_circles(n_samples=3000, factor=0.7,noise=0.1, random_state=seed),
            'iris':[load_iris().data,load_iris().target],
            'titanic':[fetch_openml(name='titanic', version=1, as_frame=True).data[['pclass','sex','age','fare']],fetch_openml(name='titanic', version=1, as_frame=True).target]}

datasets_info = {'make_moons':None,'make_circles':None,'iris':None,
                 'titanic':['continuous','categorical','continuous','continuous','continuous']}

RF_options = {'max_depth':10, 'min_group_size':20, 'n_trees':50,
              'replacement':True,'percen_L':0.8, 'percen_M':1,'seed':seed}

# Load Dataset
ds = 'titanic'
X, y = datasets[ds]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

if ds == 'titanic':
    ind = X.dropna().index
    X = X.loc[ind].values
    y = y.loc[ind].values

    y = np.array([int(yy) for yy in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    preprocessor = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1]) ], remainder='passthrough')

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

# Decision Tree
print('Decision Tree')
myTree = DecisionTree(max_depth=RF_options['max_depth'],min_group_size=RF_options['min_group_size'], random_state=RF_options['seed'])
myTree.fit(X_train,y_train,datasets_info[ds])

print('(%) Error training data:', error_rate(y_train,myTree.predict(X_train)))
print('(%) Error testing data:',error_rate(y_test,myTree.predict(X_test)))

# Random Forest
print('Random Forest')
myforest = RandomForest(percen_L = RF_options['percen_L'],percen_M=RF_options['percen_M'],
                        n_trees=RF_options['n_trees'],replacement = RF_options['replacement'],
                        max_depth=RF_options['max_depth'],min_group_size=RF_options['min_group_size'], random_state=RF_options['seed'])

df_forest = myforest.fit(X_train,y_train,datasets_info[ds])

#df_rules = get_rules(1,df_forest,X_test.copy(),y_test.copy())
#df_rules.to_csv('results/rules_'+ds+'.csv')

print('(%) Error training data:',error_rate(y_train,myforest.predict(X_train)))
print('(%) Error Testing data:',error_rate(y_test,myforest.predict(X_test)))


# ScikitLearn
print('Scikit-Learn Decision Tree')

rf = DecisionTreeClassifier( max_depth=RF_options['max_depth'],
                            min_samples_split=RF_options['min_group_size'],min_samples_leaf=RF_options['min_group_size']
                            )
rf.fit(X_train, y_train)

print('(%) Error training data:',error_rate(y_train,rf.predict(X_train)))
print('(%) Error Testing data:',error_rate(y_test,rf.predict(X_test)))

print('Scikit-Learn Random Forest')

rf = RandomForestClassifier(n_estimators = RF_options['n_trees'], random_state = 296,max_depth=RF_options['max_depth'],
                            min_samples_split=RF_options['min_group_size'],min_samples_leaf=RF_options['min_group_size'],
                            bootstrap=True, max_samples=RF_options['percen_L'])
rf.fit(X_train, y_train)

print('(%) Error training data:',error_rate(y_train,rf.predict(X_train)))
print('(%) Error Testing data:',error_rate(y_test,rf.predict(X_test)))

if ds in ['make_moons','make_circles']:
    myTree.plot_tree(X_train,y_train,True,ds)
