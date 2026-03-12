#%%
#Caj Rollny, assignment 3 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

df_combined = pd.concat([df1, df2, df3])

#1.3: Remove the columns start_time, axle, cluster, tsne_1, and tsne_2 from the dataset.
del df_combined["start_time"] 
del df_combined["axle"]
del df_combined["cluster"]
del df_combined["tsne_1"]
del df_combined["tsne_2"]


#%%
#1.4: Replace all normal events with 0 and all other events with 1.
#using vectorization, got som help here: https://www.geeksforgeeks.org/pandas/different-ways-to-iterate-over-rows-in-pandas-dataframe/
df_combined_normal = df_combined.copy()
df_combined_normal['event'] = np.where(df_combined["event"] == "normal", "1", "0")

#save resulting dataset:
df_combined_normal.to_csv("lab3_combined_dataset.csv", index=False)

# %%
#2.1:  Normalize the dataset.


