import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import Mondrian
import DifferentialPrivacy
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
infoLossMondrian = []

for k in k_values:
    infoLossMondrian.append(InformationLoss(Mondrian.mondrian(k,originalData)))

fig, ax = plt.subplots()
ax.set_ylabel('Information Loss')
ax.set_xlabel('k values')
ax.set_title('Information loss related to k values')
plt.plot(k_values,infoLossMondrian)
plt.savefig('MondrianIF.png')

# Analysis of Epsilon in Differential Privacy

epsilon_values = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370]
infoLossDiffPriv = []

for epsilons in epsilon_values:
    infoLossDiffPriv.append(InformationLoss(DifferentialPrivacy.DifferentialPrivacy(epsilons,originalData)))

fig, ax = plt.subplots()
ax.set_ylabel('Information Loss')
ax.set_xlabel('Epsilon values')
ax.set_title('Information loss related to Epsilon values')
plt.plot(epsilon_values,infoLossDiffPriv)
plt.savefig('DiffPrivIF.png')

# Analysis of k values for Optimal Univariate Microaggregation

k_values = [2,3,4,5,6,7,8,9,10,11,12,13]
infoLossOUM = []

for k in k_values:
    infoLossOUM.append(InformationLoss(OptimalUnivariateMicroaggregation.OptimalUnivariantMicroaggregation(k,originalData)))

fig, ax = plt.subplots()
ax.set_ylabel('Information Loss')
ax.set_xlabel('k values')
ax.set_title('Information loss related to k values')
plt.plot(k_values,infoLossOUM)
plt.savefig('OumIF.png')
