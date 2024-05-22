import pandas as pd
import recordlinkage
import recordlinkage.index
import recordlinkage.compare
import localDifferentialPrivacy

originalData = pd.read_csv("adaptData.csv", header=None)
DiffPrivData = localDifferentialPrivacy.localDifferentialPrivacy(50, originalData)

DiffPrivData.to_csv("proves.csv",header=None,index=None)

indexer = recordlinkage.Index()
indexer.full()
candidate_pairs = indexer.index(originalData, DiffPrivData)

comp = recordlinkage.Compare()
comp.numeric(0,0)
comp.numeric(1,1)
comp.numeric(2,2)
comp.numeric(3,3)
comp.numeric(4,4)
comp.numeric(5,5)
features = comp.compute(candidate_pairs, originalData, DiffPrivData)

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
print("Error percentage:",errors/206 * 100)


    

