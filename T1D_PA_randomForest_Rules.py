import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.random_forest import RandomForest,error_rate
from src.extract_rules import back_rule, test_rules

seed = 296
np.random.seed(seed)

features = ['f_activity_classification','f_pa_activity_duration_min','f_pa_activity_intensity_kcal_per_min',
            'f_pa_activity_time_of_day_hour_sin','f_pa_activity_time_of_day_hour_cos',
            'f_pa_time_since_prev_pa_event_hr', 'f_glucose_variability_lback_24h',
            'f_lbgi_lback_24h','f_glucose_roc_lback_30min',
            'f_glucose_start','f_iob_start', 'f_demographics_age', 'f_clinical_years_with_t1d']

datasets_info = ['categorical','continuous','continuous','continuous','continuous','continuous',
                 'continuous','continuous','continuous','continuous','continuous','continuous','continuous']

target = ['tg_time_hypo_during_pa']
df = pd.read_csv('data/mrf_tidepool_dataset_with_clusters_11072021_VRE.csv').dropna(subset=target)

for i,f in enumerate(features):
    if datasets_info[i]=='continuous':
        df.loc[df[f].isna(),f] = df[f].mean()
    else:
        df.loc[df[f].isna(),f] = df[f].mode()


X_train = df.loc[df.is_train==1,features].values
y_train = df.loc[df.is_train==1,target].values
y_train[y_train>0] = 1

X_test = df.loc[df.is_train==0,features].values
y_test = df.loc[df.is_train==0,target].values
y_test[y_test>0] = 1

print(len(X_train))
print(len(X_test))
preprocessor = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0]) ], remainder='passthrough')

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

columns = preprocessor.get_feature_names_out()

columns_before = ['feat_'+str(i)+'_<' for i in range(X_train.shape[1])]+['feat_'+str(i)+'_>=' for i in range(X_train.shape[1])]
dict_columns = {}
for c in range(len(columns)):
    info = columns[c].split('__x')[-1]
    index = info.split('_')[0]
    category = info.split('_')[-1]
    if index==category:
        dict_columns[columns_before[c]] = features[int(index)]+'_'+columns_before[c].split('_')[-1]
        dict_columns[columns_before[c+len(columns_before)//2]] = features[int(index)]+'_'+columns_before[c+len(columns_before)//2].split('_')[-1]
    else:
        dict_columns[columns_before[c]] = features[int(index)]+'_'+category+'_!='
        dict_columns[columns_before[c+len(columns_before)//2]] = features[int(index)]+'_'+category+'_='


RF_options = {'max_depth':20, 'min_group_size':5, 'n_trees':100,
              'replacement':True,'percen_L':0.8, 'percen_M':1,'seed':seed}

myforest = RandomForest(percen_L = RF_options['percen_L'],percen_M=RF_options['percen_M'],
                        n_trees=RF_options['n_trees'],replacement = RF_options['replacement'],
                        max_depth=RF_options['max_depth'],min_group_size=RF_options['min_group_size'], random_state=RF_options['seed'])

df_forest = myforest.fit(X_train,y_train)
print('(%) Error training data:',error_rate(y_train,myforest.predict(X_train)))
print('(%) Error Testing data:',error_rate(y_test,myforest.predict(X_test)))


value = 1 # Hypo

rules = []

for tree in df_forest.tree.unique():
    df_tree = df_forest.loc[df_forest.tree==tree]
    leaf_interest = df_tree.loc[df_tree.value==value]

    for i in leaf_interest.index:
        rule = [[df_tree.loc[i,'feat_index'],df_tree.loc[i,'th'],df_tree.loc[i,'side']]]
        rule = back_rule(rule,df_tree,i)
        rules.append(rule)


for i,r in enumerate(rules):
    df_rule = test_rules(X_test,y_test,r,value)

    if i==0:
        df_rules = df_rule
    else:
        df_rules = pd.concat([df_rules,df_rule])


df_rules = df_rules.rename(columns=dict_columns)
df_rules.to_csv('results/rules_T1D_PA.csv',index=False)

