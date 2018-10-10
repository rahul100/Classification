
# coding: utf-8

# In[71]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import warnings
warnings.filterwarnings('ignore')


# In[72]:

df = pd.read_csv("./h_c.csv")
df = df.replace('?', np.NaN)
df_cat_one_hot = pd.get_dummies(df[['C1','C4','C5','C6','C7','C9','C10','C12','C13']])
df_cat_rest = df[['C2','C3','C8','C11','C14','C15','Hired']]
df_new = pd.concat([df_cat_one_hot,df_cat_rest] , axis =1)
X_train, X_test, y_train, y_test = train_test_split(df_new.drop('Hired', axis=1), df_new.loc[:,['Hired']], test_size=0.20, random_state=5)


# In[73]:

print "The training data looks like below"
print df.head(2)

print "Summary of the numeric columns"
print df.describe()

print "Summary of the categorial columns"
for col in ['C4','C5','C7','C9','C10','C12','C13','Hired']:
    print "---------------\n" , col , "\n---------------"
    print df.apply(lambda x: x.value_counts(dropna = False)).T.stack().loc[col]


# In[74]:

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# # Missing Value Treatment

# The startegy is to replce the categorical variables with the most frequent value while the numeric features are replaced with mean

# In[75]:

imputer = DataFrameImputer().fit(X_train)
X_train_tf = imputer.transform(X_train)
X_test_tf = imputer.transform(X_test)
X_train_tf=X_train_tf.reset_index(drop=True)
X_test_tf=X_test_tf.reset_index(drop=True)
y_train, y_test = y_train.reset_index(drop=True) , y_test.reset_index(drop=True)


# In[76]:

X_train_tf[['C2','C14']] = X_train_tf[['C2','C14']].apply(pd.to_numeric)
X_test_tf[['C2','C14']] = X_test_tf[['C2','C14']].apply(pd.to_numeric)


# #  Randon Forest classifier

# In[77]:

# Create random forest classifier instance
clf = RandomForestClassifier()
clf.fit(X_train_tf, y_train)
print "Trained model :: ", clf
predictions = clf.predict(X_test_tf)


# In[78]:

# Train and Test Accuracy
print "Train Accuracy :: ", accuracy_score(y_train, clf.predict(X_train_tf))
print "Test Accuracy  :: ", accuracy_score(y_test, predictions)
print " Confusion matrix ", confusion_matrix(y_test, predictions)


# # XG Boost classifier

# In[79]:

xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)


# In[80]:

def modelfit(alg, dtrain, predictors,y_train,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=y_train['Hired'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], y_train['Hired'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(y_train['Hired'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y_train['Hired'], dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


# In[81]:

modelfit(xgb2, X_train_tf, X_train_tf.columns , y_train)


# In[82]:

# Test Accuracy


# In[83]:

dtest_predictions = xgb2.predict(X_test_tf)
dtest_predprob = xgb2.predict_proba(X_test_tf)[:,1]
        
#Print model report:
print "\nModel Report"
print "Accuracy (Test) : %.4g" % metrics.accuracy_score(y_test['Hired'].values, dtest_predictions)
print "AUC Score (Test): %f" % metrics.roc_auc_score(y_test['Hired'], dtest_predprob)


# ## Parameter tuning for XGBoost Model

# In[ ]:

param_test1 = {
 'max_depth':range(3,6,2),
# 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train_tf,y_train['Hired'])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# ## Fully connected Deep Net using Keras

# In[200]:

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[209]:

#Initializing Neural Network
classifier = Sequential()


# In[210]:

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 46))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# In[211]:

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[212]:

# Fitting our model 
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
history = LossHistory()
classifier.fit(X_train_tf.values, y_train.values, batch_size = 10, nb_epoch = 100,callbacks=[history])


# In[215]:

# Predicting the Test set results
y_pred = classifier.predict(X_test_tf.values)
y_pred = [1 if y>0.5 else 0 for y in y_pred]


# In[216]:

#Print model report:
print "\nFully connected NN Model Report"
print "Accuracy (Test) : %.4g" % metrics.accuracy_score(y_test['Hired'].values, y_pred)

