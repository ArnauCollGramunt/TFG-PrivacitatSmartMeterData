import pandas as pd
import recordlinkage
import recordlinkage.index
import recordlinkage.compare
import DifferentialPrivacy
import matplotlib.pyplot as plt
import Mondrian

originalData = pd.read_csv("adaptData.csv", header=None)

def RecordLinkageFunc(anonymized_data):
    indexer = recordlinkage.Index()
    indexer.full()
    candidate_pairs = indexer.index(originalData, anonymized_data)

    comp = recordlinkage.Compare()
    comp.numeric(0,0)
    comp.numeric(1,1)
    comp.numeric(2,2)
    comp.numeric(3,3)
    comp.numeric(4,4)
    comp.numeric(5,5)
    features = comp.compute(candidate_pairs, originalData, anonymized_data)

    array = []
    for index1 in range(206):
        sum = 0
        equivalent = []
        for index2 in range(206):
            new = features.loc[(index1,index2)].sum()
            if(sum < new):
                sum = new
                equivalent = [index1,index2]
        array.append(equivalent)

    errors = 0
    for x in array:
        if(x[0] != x[1]):
            errors = errors + 1
    
    return (errors/206 * 100)

# Analysis of epsilon values for Differential privacy

epsilon_values = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370]

linkage_error_percentage_DiffPriv = []

for epsilons in epsilon_values:
    linkage_error_percentage_DiffPriv.append(RecordLinkageFunc(DifferentialPrivacy.DifferentialPrivacy(epsilons,originalData)))

fig, ax = plt.subplots()
ax.set_ylabel('Linkage error percentage')
ax.set_xlabel('Epsilon values')
ax.set_title('Linkage error percentage related to Epsilon values')
plt.plot(epsilon_values,linkage_error_percentage_DiffPriv)
plt.savefig('DiffPrivRC.png')  


# Analysis of k values for Mondrian

k_values = [2,3,4,5,6,7,8,9,10,11,12,13]
linkage_error_percentage_Mondrian = []

for k in k_values:
    linkage_error_percentage_Mondrian.append(RecordLinkageFunc(Mondrian.mondrian(k,originalData)))

fig, ax = plt.subplots()
ax.set_ylabel('Linkage error percentage')
ax.set_xlabel('k values')
ax.set_title('Linkage error percentage related to k values')
plt.plot(k_values,linkage_error_percentage_Mondrian)
plt.savefig('MondrianRC.png')



