import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#from matplotlib import pyplot as plt

data = pd.read_csv("Immunotherapy.csv", delimiter=",") #import data
xData = data[['Sex','Age', 'Time', 'Number_of_Warts', 'Type', 'Area', 'Induration_Diameter']] #seperate independent variables
yData = data[['Result_of_Treatment']].values.flatten() #seperate dependent variables
feature_names = data.columns.values.tolist()
del feature_names[-1]


x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.3) #create the training set and the testing set
model = RandomForestClassifier(n_estimators = 100, bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)  #creates the random forest model with n estimators

model.fit(x_train, y_train) #fits the training data in the model to generate predictions
treatment_result=model.predict(x_test) #uses the training data to make a predictions using the test data

feature_imp = pd.Series(model.feature_importances_,index=feature_names).sort_values(ascending=False)
print(feature_imp)
print("Accuracy:",metrics.accuracy_score(y_test, treatment_result))

#remove unimportant features to increase accuracy
