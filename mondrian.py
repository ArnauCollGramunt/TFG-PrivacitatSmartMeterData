import pandas as pd
import numpy as np
import sys

k = int(sys.argv[1])
data = pd.read_csv(sys.argv[2], header=None)

def mondrian():
    ranks={}

    for columns in range(len(data.columns)):
        ranks[columns] = len(set(data.iloc[:, columns]))
    
    ranks = sorted(ranks.items(), key=lambda t: t[1], reverse=True)
    return mondrianRec(data,ranks[0][0])

def mondrianRec(partition, dim):
    partition = partition.sort_values(by=partition.columns[dim])
    
    num = partition.iloc[:, dim].size

    mid = num // 2

    lhs = partition[:mid]
    rhs = partition[mid:]

    if(len(lhs) >= k and len(rhs) >= k):
        return pd.concat([mondrianRec(lhs,dim),mondrianRec(rhs,dim)])
    
    for columns in range(len(data.columns)):
        partition = partition.sort_values(by=partition.columns[dim])
        new = partition.iloc[:, columns].mean()
        partition.iloc[:, columns] = new

    return partition

example = mondrian()

example.to_csv("example.csv",index=False)