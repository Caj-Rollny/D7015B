#%%
#Caj Rollny, assignment 3 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

'''
Task to pass with Grade 3:
1.	Data preprocessing:
    1: Load the data from all three files.
    2: Combine the three datasets into a single unified dataset.
    3: Remove the columns start_time, axle, cluster, tsne_1, and tsne_2 from the dataset.
    4: Replace all normal events with 0 and all other events with 1.
2.	Data transformation:
    1: Normalize the dataset.
    
'''
#%%
# 1.1_ read the data set into dataframes
df1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1-1-1.csv') 
df2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1-2.csv')
df3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0-2.csv')
# 1.2: Combine the datasets:

df_combined = pd.concat([df1, df2, df3], ignore_index=True)

#1.3: Remove the columns start_time, axle, cluster, tsne_1, and tsne_2 from the dataset.
del df_combined["start_time"] 
del df_combined["axle"]
del df_combined["cluster"]
del df_combined["tsne_1"]
del df_combined["tsne_2"]

#print(df_combined['event'].value_counts())

#%%
#1.4: Replace all normal events with 0 and all other events with 1.
#using vectorization, got som help here: https://www.geeksforgeeks.org/pandas/different-ways-to-iterate-over-rows-in-pandas-dataframe/
df_combined_normal = df_combined.copy()
df_combined_normal['event'] = np.where(df_combined["event"] == "normal", "0", "1")

#save resulting dataset:
df_combined_normal.to_csv("lab3_combined_dataset.csv", index=False)

# %%
#2.1:  Normalize the dataset. 
#check if NULL/NaN values in dataset:
#print(df_combined_normal.isnull().sum())
#dataset looks fine.
#separate features and target:
#features:
X = df_combined_normal.copy()
#remove target from features:
del X["event"]
#target:
y = df_combined_normal["event"]
#normalize features: From the support-lecture: Standardize features to zero mean, unit variance (using StandardScaler)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#print(X.mean())
#print(X_scaled.mean())


#%%

'''
Task to pass with Grade 4:
1.	Dataset splitting:
Split the data into training and testing sets in an 80/20 ratio.
2.	Cross-Validation:
Perform k-fold cross-validation (e.g., 5-fold) on the training set to evaluate model stability.
3.	Comparison task: Compare between the 80/20 train-test split and k-fold cross-validation using SVM (Support Vector Machine).  Train an SVM model using both methods and evaluate its performance. Discuss the differences in accuracy, consistency of results, and generalization ability.
'''
#1: Split the data into training and testing sets in an 80/20 ratio:
# ref: support lecture and https://stackoverflow.com/questions/73475197/how-to-implement-machine-learning-model-svc-using-training-validation-and-test

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import accuracy_score, classification_report
#Split data: 80% train, 20% test:
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2, random_state=42)

#Train SVM classifier:
svm_model = svm.SVC()
svm_model.fit(X_train,y_train)
#Evaluate:
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nSVM Accuracy (80/20 plit): {accuracy:.4f} ')
print(classification_report(y_test,y_pred,target_names=['Normal', 'Event']))

#%%
#2: k-fold cross-validation, 5-fold, om training dataset:
cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=True)
#Perform cross-validation:
svm_cv = svm.SVC(kernel='rbf',random_state=42)
cv_scores = cross_val_score(svm_cv,X_scaled,y,cv=cv,scoring='accuracy')
print("K-Fold Cross-ValidationResults 5-Fold\n")
for i, score in enumerate(cv_scores, 1):
    print(f" Fold{i}: {score:.4f}")

print(f'\nMean Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')


# %%
