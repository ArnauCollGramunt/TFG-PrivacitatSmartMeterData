import pandas as pd
import numpy as np

def DifferentialPrivacy(epsilon,data):
    new_df = pd.DataFrame()

    for column in range(len(data.columns)):
        n = data[column].count()
        sensitivity = abs(data[column].max() - data[column].min())
        b = sensitivity / epsilon
        new_values = np.random.laplace(0,b,n)
        new_df[column] = new_values + data.iloc[:, column]

    return(new_df)