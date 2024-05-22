import pandas as pd

with open('adaptData.csv', 'w') as archivo:
    for i in range(168):
        df = pd.read_csv(f"../datasets/Small LCL Data/LCL-June2015v2_{i}.csv", sep=",")
        for lclid in df['LCLid'].unique():
            df_temp = df[df['LCLid'] == lclid].head(48)
            kwh = df_temp.iloc[:, 3].tolist() 
            archivo.write(','.join(map(str, kwh)) + '\n')


