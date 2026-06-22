def QuanQual(st_habits1):
    qual=[]
    quan=[]
    for columnname in st_habits1.columns:
        # print(columnname)
        if(st_habits1[columnname].dtype=='O'):
            qual.append(columnname)
        else:
            quan.append(columnname)
    print ("Qualitative columns = ",qual)
    print ("Quantitative columns = ",quan)
    return quan,qual