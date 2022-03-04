import os
import re 
import numpy as np

def line2float(line):
    data=list()
    newstr=str()
    data.append(newstr)
    index=0
    for i in line:
        if((i<='9'and i>='0') or i =='.'):
           data[index]+=i
        elif(i ==','):
            index+=1
            newstr=str()
            data.append(newstr)
    for i in range(len(data)):
        data[i]=float(data[i])
    return data
         