import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import mondrian
import localDifferentialPrivacy
import OptimalUnivariateMicroaggregation

originalData = pd.read_csv("adaptData.csv",header=None)

def InformationLoss(maskedData):
    sum = 0
    if 'row_id' in originalData.columns:
        originalData.drop(columns=['row_id'],inplace=True)
    for column in range(len(originalData.columns)):
        stndrDev = statistics.stdev(originalData.iloc[:, column])
        sum = np.sum(abs(originalData.iloc[:, column] - maskedData.iloc[:, column]) / math.sqrt(2*stndrDev)) + sum
    infoLoss = (1 / (48 * originalData.shape[0])) * sum
    return infoLoss

# Analysis of k values for Mondrian

k_values = [2,3,4,5,6,7,8,9,10,11,12,13]
infoLoss = []

for k in k_values:
    infoLoss.append(InformationLoss(mondrian.mondrian(k,originalData)))

fig, ax = plt.subplots()
ax.set_ylabel('Information Loss')
ax.set_xlabel('k values')
ax.set_title('Information loss related to k values')
plt.bar(k_values,infoLoss)
plt.savefig('fig.png')

# Analysis of Epsilon in Differential Privacy

epsilon_values = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50]
infoLoss = []

for epsilons in epsilon_values:
    infoLoss.append(InformationLoss(localDifferentialPrivacy.localDifferentialPrivacy(epsilons,originalData)))

fig, ax = plt.subplots()
ax.set_ylabel('Information Loss')
ax.set_xlabel('Epsilon values')
ax.set_title('Information loss related to Epsilon values')
plt.bar(epsilon_values,infoLoss)
plt.savefig('fig2.png')

# Analysis of k values for Optimal Univariate Microaggregation

k_values = [2,3,4,5,6,7,8,9,10,11,12,13]
infoLoss = []

for k in k_values:
    infoLoss.append(InformationLoss(OptimalUnivariateMicroaggregation.OptimalUnivariantMicroaggregation(k,originalData)))

fig, ax = plt.subplots()
ax.set_ylabel('Information Loss')
ax.set_xlabel('k values')
ax.set_title('Information loss related to k values')
plt.bar(k_values,infoLoss)
plt.savefig('fig3.png')