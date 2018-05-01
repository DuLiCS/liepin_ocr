import numpy as np


# The range of the input binary img's pixels is from 0 to 1


def xylimit(pic):
    m = pic.shape[0]
    n= pic.shape[1]

    Ycount = np.zeros([1,m])
    for i in range(0,m):
        Ycount[0,i] = np.sum(pic[i,0:n])
    Ybottom = m-1
    Yvalue = Ycount[0,Ybottom]
    while(Yvalue<3):
        Ybottom = Ybottom-1
        Yvalue = Ycount[0,Ybottom]
    Yceil = 0
    Yvalue = Ycount[0,Yceil]
    while(Yvalue<3):
        Yceil = Yceil+1
        Yvalue = Ycount[0,Yceil]
    Xcount = np.zeros([1,n])
    for j in range(0,n):
        Xcount[0,j] = np.sum(pic[0:m,j])

    Xleft  =0
    Xvalue = Xcount[0,Xleft]

    while(Xvalue<2):
        Xleft =  Xleft+1
        Xvalue = Xcount[0,Xleft]
    Xright = n-1
    Xvalue = Xcount[0,Xright]
    while(Xvalue<2):
        Xright = Xright-1
        Xvalue = Xcount[0,Xright]
    newpic = pic[Yceil:Ybottom,Xleft:Xright]
    return newpic