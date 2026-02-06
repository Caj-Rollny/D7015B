#%%

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


'''
Submit your solutions in pdf format, with code and plots supporting your answers.
machine_data contains raw data of a part from 3 manufactures A, B, C
The system is run to failure under load
The load and the operation time is provided in each row

What is the range of load and time during operation for each manufacturer?
What is the most expected load value?
How are the load and time related?
Which distribution best describes the load?
Which distribution best describes the time?

Which manufacturer has the best performance and why?

'''
#%%
# read the data file into a dataframe
df = pd.read_csv('machine_data.csv', index_col=0) #CajRollny: dropped the index column when reading csv file

print(df.shape)

"""
Drop the index
Caj Rollny: See comment above - dropped it when reading the csv-file.
"""

#%%
"""
Extract data for a given manufacturer
"""
grpByManu = df.groupby(['manufacturef'])

dfa = grpByManu.get_group('A')
#Caj Rollny: compare all 3 manufacturers
dfb = grpByManu.get_group('B')
dfc = grpByManu.get_group('c')


#%%

loada = dfa['load']
timea = dfa['time']
#Caj Rollny: compare all 3 manufacturers
loadb = dfb['load']
timeb = dfb['time']
loadc = dfc['load']
timec = dfc['time']

#%%
'''
Is there a relationship between load and time
'''
#Caj Rollny:generate graph:s for all manufacturers
plt.scatter(loada, timea, color='blue')
plt.title("Relation between load and time, manuf. A")
plt.xlabel("Load")
plt.ylabel("Time")
plt.show()

plt.scatter(loadb, timeb, color='green')
plt.title("Relation between load and time, manuf. B")
plt.xlabel("Load")
plt.ylabel("Time")
plt.show()

plt.scatter(loadc, timec, color='red')
plt.title("Relation between load and time, manuf. C")
plt.xlabel("Load")
plt.ylabel("Time")
plt.show()

#Caj Rollny: switch to plot load over time:
plt.scatter(timea, loada, color='blue')
plt.title("Relation between load and time, manuf. A")
plt.xlabel("Time")
plt.ylabel("Load")
plt.show()

#Caj Rollny: plot all in one graph:


#%%
'''
Characteristics of data
mean, median, mode
'''
print("Mean A:", dfa['load'].mean())
print("Mode A:", dfa['load'].mode())
print("Median A:", dfa['load'].median(), "\n")
print("Mean B:", dfb['load'].mean())
print("Mode B:", dfb['load'].mode())
print("Median B:", dfb['load'].median(), "\n")
print("Mean c:", dfc['load'].mean())
print("Mode c:", dfc['load'].mode())
print("Median c:", dfc['load'].median())

#%%
'''
How is load distributed
Why does it matter
uniform, normal, exponential, weibull
'''
dfa[['load']].plot(kind='hist', bins=10)
dfb[['load']].plot(kind='hist', bins=10)
dfc[['load']].plot(kind='hist', bins=10)


#%%
'''
variance, standard deviation
What is the meaning of 6sigma
'''
#%%
'''
Other plots that can be useful 
boxplot
'''
#Caj Rollny: let's look at a boxplot compairing all manuf:
all_manuf = [dfa, dfb, dfc]
labels = ['Manyf A', 'Manuf B', 'Manuf c']
colors = ['peachpuff', 'orange', 'tomato']
fig, ax = plt.subplots()
ax.set_ylabel('Load')
bplot = ax.boxplot(all_manuf,
                   patch_artist=True,  # fill with color
                   tick_labels=labels)  # will be used to label x-ticks

# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.show()
# %%
