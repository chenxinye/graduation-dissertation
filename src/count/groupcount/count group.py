import pandas as pd

for i in [i + 1 for i in range(25)]:    
    if i == 1:
        df = pd.read_csv("topic"+" Top" + str(i) + " .csv")[["word","n"]]
        df.columns = ["top1","count1"]
    else:
        df_ = pd.read_csv("topic"+" Top" + str(i) + " .csv")[["word","n"]]
        df_.columns = ["top"+str(i),"count" + str(i)]
        df = pd.concat([df,df_],axis = 1)
df.to_csv("df.csv",index = 0)