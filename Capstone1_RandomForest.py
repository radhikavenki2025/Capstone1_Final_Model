import pandas as pd
from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
import shap

shap.initjs()

def selectkbest(indep_X,dep_Y,n):
    test = SelectKBest(score_func=chi2, k=n)
    fit1= test.fit(indep_X,dep_Y)
    selectk_features = fit1.transform(indep_X)
    return selectk_features


def split_scalar(indep_X,dep_Y):
    X_train,X_test,y_train,y_test = train_test_split(indep_X, dep_Y , test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform (X_test)
    return X_train , X_test , y_train , y_test

def split_unscaled(indep_X, dep_Y):
    
    X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size=0.25, random_state=0 )
                                                        
    return X_train, X_test, y_train, y_test

    

def cm_prediction(classifier,X_test ,y_test):
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix ,accuracy_score , classification_report
    cm = confusion_matrix(y_test , y_pred)

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    
    Accuracy = accuracy_score(y_test , y_pred)

    report = classification_report(y_test, y_pred)
    return Accuracy,report,cm


def logistic(X_train,y_train,X_test,y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train , y_train)
    Accuracy,report,cm = cm_prediction(classifier,X_test,y_test)
    return classifier,Accuracy,report,cm

def svm_linear(X_train,y_train,X_test,y_test):

    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear' , random_state = 0)
    classifier.fit(X_train , y_train)
    Accuracy, report,cm = cm_prediction(classifier,X_test,y_test)
    return classifier,Accuracy,report,cm
    
def svm_NL(X_train, y_train,X_test,y_test):

    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf',random_state = 0 )
    classifier.fit(X_train,y_train)
    Accuracy,report,cm = cm_prediction(classifier,X_test,y_test)
    return classifier,Accuracy,report,cm

def Navie(X_train,y_train,X_test,y_test):

    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train , y_train)
    Accuracy,report,cm = cm_prediction(classifier,X_test,y_test)
    return classifier,Accuracy,report,cm
    
def knn(X_train,y_train,X_test,y_test):
       
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        Accuracy,report,cm = cm_prediction(classifier,X_test,y_test)
        return  classifier,Accuracy,report,cm

def Decision(X_train,y_train,X_test,y_test):

        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        Accuracy,report,cm = cm_prediction(classifier,X_test,y_test)
        return  classifier,Accuracy,report,cm
    
def random(X_train,y_train,X_test,y_test):
    
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',random_state = 0)
        classifier.fit(X_train, y_train)
        Accuracy,report,cm = cm_prediction(classifier,X_test,y_test)
        return  classifier,Accuracy,report,cm

def selectk_Classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf): 
    
    dataframe=pd.DataFrame(index=['ChiSquare'],columns=['Logistic','SVMl','SVMnl','KNN','Navie','Decision','Random'])
    
    for number,idex in enumerate(dataframe.index):      
        dataframe['Logistic'][idex]=acclog[number]       
        dataframe['SVMl'][idex]=accsvml[number]
        dataframe['SVMnl'][idex]=accsvmnl[number]
        dataframe['KNN'][idex]=accknn[number]
        dataframe['Navie'][idex]=accnav[number]
        dataframe['Decision'][idex]=accdes[number]
        dataframe['Random'][idex]=accrf[number]
    return dataframe

def random_full(X_train, y_train, X_test, y_test):
   
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    Accuracy, report, cm = cm_prediction(classifier, X_test, y_test)
    return classifier, Accuracy, report, cm

def shap_explain(classifier, X_test, feature_names=None, sample_index=0):

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        at_risk = shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        at_risk = shap_values[:, :, 0]
    else:
        at_risk = shap_values

  
    cols = feature_names if feature_names is not None else (X_test.columns if hasattr(X_test, 'columns') else None )

    
    print("Generating SHAP summary plot (global feature impact)...")
    shap.summary_plot(at_risk, X_test, feature_names=cols)

    print("\nGenerating SHAP bar plot (simplified ranking)...")
    shap.summary_plot(at_risk, X_test, feature_names=cols, plot_type='bar')

    expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    row = X_test.iloc[sample_index] if hasattr(X_test, 'iloc') else X_test[sample_index]

    print(f"\nGenerating force plot for student at index {sample_index}...")
    force_plot = shap.force_plot( expected_value, at_risk[sample_index], row, feature_names=cols)
       

    return shap_values, explainer
    