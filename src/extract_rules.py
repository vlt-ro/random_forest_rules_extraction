import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def back_rule(rule, df_tree,i):
    parent_index = df_tree.loc[i,'parent_feat_index']
    parent_th = df_tree.loc[i,'parent_th']

    index_parent = df_tree.loc[np.logical_and(df_tree.feat_index==parent_index,df_tree.th==parent_th)].index[0]

    if df_tree.loc[index_parent,'side']=='root':
        rule.append([parent_index,parent_th,'root'])
        return rule
    else:
        rule.append([parent_index,parent_th,df_tree.loc[index_parent,'side']])
        rule = back_rule(rule,df_tree,index_parent)
        return rule

def test_rules(X,y,rule,value):
    df_data = pd.DataFrame(X)
    y[y!=value]=-1
    df_data['y_true'] = y
    df_data['y_pred'] = -1
    df_rule = pd.DataFrame(columns=['feat_'+str(i)+'_<' for i in range(X.shape[1])]+['feat_'+str(i)+'_>=' for i in range(X.shape[1])],index=[0])
    rule.reverse()
    sub_X = df_data.copy()
    for i,r in enumerate(rule[:-1]):

        index = r[0]
        th = r[1]
        side = rule[i+1][2]
        if side=='left':
            df_rule.loc[0,'feat_'+str(index)+'_<'] = th

            sub_X = sub_X.loc[sub_X[index]<th]
        elif side=='right':
            df_rule.loc[0,'feat_'+str(index)+'_>='] = th

            sub_X = sub_X.loc[sub_X[index]>=th]

    df_data.loc[sub_X.index,'y_pred'] = 0
    tn, fp, fn, tp = confusion_matrix(df_data.y_true,df_data.y_pred,labels=[value,-1]).ravel()
    df_rule['precision'] = tp/(tp+fp)
    df_rule['recall'] = tp/(fn+tp)
    df_rule['accuracy']  =  (tp + tn)/ (tp + fn + tn + fp)
    return df_rule

def get_rules(value,df_forest,X_test,y_test):
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

    return df_rules