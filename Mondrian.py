import pandas as pd
import numpy as np

def mondrian(k,data):
    data['row_id'] = range(len(data))
    ranks={}

    for columns in range(len(data.columns) - 1):
        ranks[columns] = len(set(data.iloc[:, columns]))
    
    ranks = sorted(ranks.items(), key=lambda t: t[1], reverse=True)

    maskedData = mondrianRec(data,ranks[0][0],k, data)
    maskedData.sort_values(by='row_id')
    maskedData.drop(columns=['row_id'],inplace=True)
    return maskedData

def mondrianRec(partition, dim, k, data):
    partition = partition.sort_values(by=partition.columns[dim])
    
    num = partition.iloc[:, dim].size

    mid = num // 2

    lhs = partition[:mid]
    rhs = partition[mid:]

    if(len(lhs) >= k and len(rhs) >= k):
        return pd.concat([mondrianRec(lhs,dim,k,data),mondrianRec(rhs,dim,k,data)])
    
    for columns in range(len(data.columns) - 1):
        partition = partition.sort_values(by=partition.columns[dim])
        new = partition.iloc[:, columns].mean()
        partition.iloc[:, columns] = new

    return partition
