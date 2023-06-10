import pandas as pd
import numpy as np

ds = 't1d_pa'

if ds == 'iris':
    df_rules = pd.read_csv('results/rules_iris.csv')
    df_rules['F1'] = 2*(df_rules.precision*df_rules.recall)/(df_rules.precision+df_rules.recall)
    th = 0.95
    df_rules = df_rules[df_rules.F1>th]

    df_final_rules = pd.DataFrame(columns=['feat_'+str(i)for i in range(4)],index=['min','max'])

    for c in df_rules.columns[1:-4]:
        ind = c.split('_')[-2]
        if  c.split('_')[-1]=='<':
            df_final_rules.loc['max','feat_'+ind] = df_rules[c].max()
        elif  c.split('_')[-1]=='>=':
            df_final_rules.loc['min','feat_'+ind] = df_rules[c].min()

    print(df_final_rules)


elif ds == 't1d_pa':
    df_rules = pd.read_csv('rules_T1D_PA.csv')
    print(len(df_rules))
    print(df_rules.loc[250].dropna())