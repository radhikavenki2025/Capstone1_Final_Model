import pandas as pd
import numpy as np
import quanqual



def Univariate(st_habits1 , quan ):

    df_describe = st_habits1.describe()
    df_describe = pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","99%","Q4:100%","IQR",
                                    "1.5rule","Lesser","Greater","Min","Max","kurtosis","skew","Var","Std"],columns=quan)
    for columnname in quan:
        df_describe.loc["Mean",columnname]     = st_habits1[columnname].mean()
        df_describe.loc["Median",columnname]   = st_habits1[columnname].median()
        df_describe.loc["Mode",columnname]     = st_habits1[columnname].mode()[0]
        df_describe.loc["Q1:25%",columnname]   = st_habits1[columnname].quantile(0.25)
        df_describe.loc["Q2:50%",columnname]   = st_habits1[columnname].quantile(0.50)
        df_describe.loc["Q3:75%",columnname]   = st_habits1[columnname].quantile(0.75)
        df_describe.loc["99%",columnname]      = np.percentile(st_habits1[columnname],99)
        df_describe.loc["Q4:100%",columnname]  = st_habits1[columnname].quantile(1.00)
        df_describe.loc["IQR",columnname]      = df_describe[columnname]["Q3:75%"]- df_describe[columnname]["Q1:25%"]
        df_describe.loc["1.5rule",columnname]  = 1.5*df_describe[columnname]["IQR"]
        df_describe.loc["Lesser",columnname]   = df_describe[columnname]["Q1:25%"]- df_describe[columnname]["1.5rule"]
        df_describe.loc["Greater",columnname]  = df_describe[columnname]["Q3:75%"]+ df_describe[columnname]["1.5rule"]
        df_describe.loc["Min",columnname]      = st_habits1[columnname].min()
        df_describe.loc["Max",columnname]      = st_habits1[columnname].max()
        df_describe.loc["kurtosis",columnname] = st_habits1[columnname].kurtosis()
        df_describe.loc["skew",columnname]     = st_habits1[columnname].skew()
        df_describe.loc["Var",columnname]      = st_habits1[columnname].var()
        df_describe.loc["Std",columnname]      = st_habits1[columnname].std()
    
    return df_describe

    