import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)

warnings.filterwarnings("ignore")

print("Версия LightGBM      : ", lgb.__version__)
print("Версия Scikit-Learn  : ", sklearn.__version__)
seed = 42

################################
########## DATA PREP ###########
################################

def get_lgbm_score(model, y_true, X):
    y_pred = model.predict(X)

    Accuracy = accuracy_score(y_true, y_pred)
    F1_score = f1_score(y_true, y_pred, average='weighted')
    Precision = precision_score(y_true, y_pred, average='weighted')
    Recall = recall_score(y_true, y_pred, average='weighted')
    print('Accuracy', round(Accuracy, 2))
    print('F1_score', round(F1_score, 2))
    print('Precision', round(Precision, 2))
    print('Recall', round(Recall, 2))   
    print()
    return Accuracy, F1_score, Precision, Recall

# Load in the data
df = pd.read_csv("./data/wine_quality.csv")

# Split into train and test sections
y = df.pop("quality")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=seed)

#################################
########## MODELLING ############
#################################

# build the lightgbm model
clf = lgb.LGBMClassifier(random_state=seed, learning_rate=0.05,)
clf.fit(X_train, y_train)

# predict the results
print('train score:')
get_lgbm_score(clf, y_train, X_train)
print('test score:')
get_lgbm_score(clf, y_test, X_test)


param_test = {
 'class_weight': ['balanced'],
 'num_leaves': [i for i in range(5, 20, 2)],
 'reg_lambda': [0, 0.001, 0.01, 0.1],
 'reg_alpha': [0, 0.1, 0.5, 5, 10],
 'n_estimators': [50, 100, 200]

}
gsearch = GridSearchCV(estimator = LGBMClassifier(n_jobs=-1, random_state=seed), 
                       param_grid = param_test, scoring='f1_weighted', n_jobs=-1, cv=5)
gsearch.fit(X_train, y_train)
gsearch.best_params_, gsearch.best_score_

print('train score:')
get_lgbm_score(gsearch.best_estimator_, y_train, X_train)
print('test score:')
Accuracy, F1_score, Precision, Recall = get_lgbm_score(gsearch.best_estimator_, y_test, X_test)

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write('Test scores:\n')
        outfile.write(f"Accuracy = {Accuracy*100}%\n")
        outfile.write(f"F1_score = {F1_score}\n")
        outfile.write(f"Precision = {Precision}\n")
        outfile.write(f"Recall = {Recall}\n")

##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################

df_feature_importance = (
    pd.DataFrame({
        'feature': X_train.columns,
        'importance': gsearch.best_estimator_.feature_importances_,
    })
    .sort_values('importance', ascending=False)
)
#df_feature_importance

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X_train.columns)), columns=['Value','Feature'])

plt.figure(figsize=(12, 5))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances.png', dpi=120)
