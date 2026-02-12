#%%
#Caj Rollny, assignment 2 
import numpy as np
import pandas as pd
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
Caj Rollny: See comment above - dropped it when reading the csv-file via the argument "index_col=0".
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
loadAll = df['load']
timeAll = df['time']


#Caj Rollny: Question 1: "What is the range of load and time during operation for each manufacturer?"

#Range of loads:
print("Manufacturer A; load min:", min(loada), "load max:", max(loada), "\n")
print("Manufacturer B; load min:", min(loadb), "load max:", max(loadb), "\n")
print("Manufacturer c; load min:", min(loadc), "load max:", max(loadc) )
#Range om times:
print("Manufacturer A; time min:", min(timea), "time max:", max(timea), "\n")
print("Manufacturer B; time min:", min(timeb), "time max:", max(timeb), "\n")
print("Manufacturer c; time min:", min(timec), "time max:", max(timec) )


#%%

#Caj Rollny: Question 2: what is the most expected load value?
#print("Mode (all):", loadAll.mode())
print("Mean (all):", loadAll.mean())
#print("Mode A:", loada.mode())
print("Mean A:", loada.mean())
#print("Mode B:", loadb.mode())
print("Mean B:", loadb.mean())
#print("Mode c:", loadc.mode())
print("Mean c:", loadc.mean())

#%%
#Caj Rollny: Question 3: "How are the load and time related?"

plt.title("Load vs time")
plt.xlabel("Load")
plt.ylabel("Time")
plt.scatter(loada,timea,color='blue', label="Manuf. A")
plt.scatter(loadb,timeb,color='red', label="Manuf. B")
plt.scatter(loadc,timec,color='green', label="Manuf. c")
plt.legend()

#%%
#Caj Rollny: Question 4 "Which distribution best describes the load?"
#Plot histograms:
dfa[['load']].plot(kind='hist', bins=10)
dfb[['load']].plot(kind='hist', bins=10)
dfc[['load']].plot(kind='hist', bins=10)

df[['load']].plot(kind='hist', bins=10)
#They look similar, "normal" distribution but skewed right, maybe "lognormal" or "Weibull"?? 
#let's try to fit the data to the probable distributions, asked ChatGPT to help me with the code!!! 
# "Load Distribution with Normal, Lognormal, Weibull Fits"
# 
import seaborn as sns
from scipy.stats import norm, lognorm, weibull_min
    
plt.figure(figsize=(10, 6))

# Histogram + KDE
sns.histplot(loadAll, bins=40, stat="density", kde=True, color="lightgray", label="Data")

# Fit Normal
mu, sigma = norm.fit(loadAll)
x = np.linspace(min(loadAll), max(loadAll), 500)
plt.plot(x, norm.pdf(x, mu, sigma), label=f"Normal (μ={mu:.2f}, σ={sigma:.2f})", linewidth=2)

# Fit Lognormal
shape, loc, scale = lognorm.fit(loadAll, floc=0)
plt.plot(x, lognorm.pdf(x, shape, loc=loc, scale=scale), label=f"Lognormal (shape={shape:.2f})", linewidth=2)

# Fit Weibull
c, loc, scale = weibull_min.fit(loadAll, floc=0)
plt.plot(x, weibull_min.pdf(x, c, loc=loc, scale=scale), label=f"Weibull (shape={c:.2f})", linewidth=2)
plt.title("Load Distribution with Normal, Lognormal, Weibull Fits")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#%%
#Caj Rollny: Question 5 "Which distribution best describes the time?"
#Plot histograms:

dfa[['time']].plot(kind='hist', bins=10)
dfb[['time']].plot(kind='hist', bins=10)
dfc[['time']].plot(kind='hist', bins=10)

df[['time']].plot(kind='hist', bins=10)
#They look similar, "normal" distribution but skewed right, maybe "lognormal" or "Weibull"?? 
#let's try to fit the data to the probable distributions, asked ChatGPT to help me with the code!!! 
# "Load Distribution with Normal, Lognormal, Weibull Fits"
# 
import seaborn as sns
from scipy.stats import norm, lognorm, weibull_min
    
plt.figure(figsize=(10, 6))

# Histogram + KDE
sns.histplot(timeAll, bins=40, stat="density", kde=True, color="lightgray", label="Data")

# Fit Normal
mu, sigma = norm.fit(timeAll)
x = np.linspace(min(timeAll), max(timeAll), 500)
plt.plot(x, norm.pdf(x, mu, sigma), label=f"Normal (μ={mu:.2f}, σ={sigma:.2f})", linewidth=2)

# Fit Lognormal
shape, loc, scale = lognorm.fit(timeAll, floc=0)
plt.plot(x, lognorm.pdf(x, shape, loc=loc, scale=scale), label=f"Lognormal (shape={shape:.2f})", linewidth=2)

# Fit Weibull
c, loc, scale = weibull_min.fit(timeAll, floc=0)
plt.plot(x, weibull_min.pdf(x, c, loc=loc, scale=scale), label=f"Weibull (shape={c:.2f})", linewidth=2)
plt.title("Time Distribution with Normal, Lognormal, Weibull Fits")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%
#Caj Rollny: Question 6: "Which manufacturer has the best performance and why?
#Plot:

plt.title("Performance comparison")
plt.xlabel("Load")
plt.ylabel("Time")
plt.scatter(loada,timea,color='blue', label="Manuf. A")
plt.scatter(loadb,timeb,color='red', label="Manuf. B")
plt.scatter(loadc,timec,color='green', label="Manuf. c")
plt.legend()

plt.show()

#Caj Rollny: calculate the mean time to failure (MTTF):

# Compute mean time to failure per manufacturer 
mean_ttf = df.groupby("manufacturef")["time"].mean() 
print(mean_ttf)
