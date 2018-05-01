import cv2

import numpy as np

from img_process import veri_seg

from xylimit import xylimit

import rotate_about_center


import matplotlib.pyplot as plt






# veri_seg('s_8.png')
#
# cv2.destroyAllWindows()
#
img = cv2.imread('b1.png')
#
# img_veri = cv2.imread('s_1.png')
#
grey_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

# grey_img_veri = cv2.cvtColor(img_veri,cv2.COLOR_RGB2GRAY)
#
binary_img = cv2.threshold(grey_img,15,1,cv2.THRESH_BINARY_INV)[1]

pic = xylimit(binary_img)


# pic = cv2.threshold(pic,0,255,cv2.THRESH_BINARY)[1]

m = pic.shape[0]
n = pic.shape[1]
Ycount = np.zeros([1,m])

for i in range(0,m):
    Ycount[0,i] = np.sum(pic[i,0:n])
lenYcount = Ycount.shape[1]
Yflag = np.zeros([1,lenYcount])

print (Ycount[0,1])

for k in range(0,lenYcount-2):
    if(Ycount[0,k]<3 and Ycount[0,k+1]<3 and Ycount[0,k+2]<3):
        Yflag[0,k] = 1
        


Yflag2 = np.zeros([1,m])
for j in range(0,m-1):
    Yflag2[0,j+1] = Yflag[0,j]

Yflag = Yflag.transpose()
Yflag2 = Yflag2.transpose()
Yflag3 = abs(Yflag-Yflag2)
[row,col]  = np.where(Yflag3 == 1)

row0 = np.zeros([1,row.shape[0]+2])
row0[0,0] = 1
for i in range(0,row.shape[0]):
    row0[0,i+1] = row[i]
row0[0,row.shape[0]+1] = m
len = int(row0.shape[1]/2)
row1 =np.zeros([1,len],int)
row2 = np.zeros([1,len],int)
for k in range(0,row0.shape[1]):
    if np.mod(k+1,2)==1:
        row1[0,int(k/2)] = row0[0,k]
    else:
        row2[0,int((k-1)/2)] = row0[0,k]

pic2 = pic[row1[0,0]:row2[0,0],]

#pic2 = cv2.threshold(pic2,0,255,cv2.THRESH_BINARY)[1]



pic = xylimit(pic2)
n = pic.shape[1]



Xcount = np.zeros([1,n])

for j in range(0,n):
    Xcount[0,j] = np.sum(pic[0:n,j])
lenXcount = Xcount.shape[1]
Xflag = np.zeros([1,lenXcount])



for k in range(0,lenXcount-2):
    if(Xcount[0,k]<3 and Xcount[0,k+1]<3 and Xcount[0,k+2]<3):
        Xflag[0,k] = 1

Xflag2 = np.zeros([1,n])
for j in range(0,n-1):
    Xflag2[0,j+1] = Xflag[0,j]

Xflag = Xflag.transpose()
Xflag2 = Xflag2.transpose()

Xflag3 = abs(Xflag-Xflag2)
[col,a]  = np.where(Xflag3 == 1)

col0 = np.zeros([1,col.shape[0]+2])
col0[0,0] = 1
for i in range(0,col.shape[0]):
    col0[0,i+1] = col[i]

col0[0,col.shape[0]+1] = n

l = col0.shape[1]

coltemp = col0[0,1:l-1]-col0[0,0:l-2]

tem = np.zeros([1,coltemp.shape[0]],int)

for i in range(0,coltemp.shape[0]):
    tem[0,i] = coltemp[i]

[b,ind] = np.where(tem < 3)

col0[0,ind] = 0
col0[0,ind+1] = 0

len1 = int(col0.shape[1]/2)


col1 = np.zeros([1,len1],int)
col2 = np.zeros([1,len1],int)
#
picnum2=int(col0.shape[1]/2)
 


for k in range(0,col0.shape[1]):
    if np.mod(k+1,2)==1:
        col1[0,int(k/2)] = col0[0,k]
    else:
        col2[0,int((k-1)/2)] = col0[0,k]

pic3 = pic[0:pic.shape[0],col1[0,0]:col2[0,0]]

pic3 = cv2.threshold(pic3,0,255,cv2.THRESH_BINARY)[1]








  
#img = plt.imread('photo.jpg')
img = pic3 
  #根据公式转成灰度图
#img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
 




fft2 = np.fft.fft2(img)

shift2center = np.fft.fftshift(fft2)
plt.imshow(np.abs(shift2center),'gray')

#plt.savefig('tempic.png')

charpic = np.abs(shift2center)


#charpic = cv2.cvtColor(charpic,cv2.COLOR_RGB2GRAY)

#a  = int(charpic.shape[0])
#
#b = int(charpic.shape[1])
#
#c =int((a - img.shape[0])/2)
#
#d = int((b - img.shape[1])/2)
#
#mid_a = int (a/2)
#
#mid_b = int (b/2)

 
#charpic_ = charpic[(mid_a-c):(a-mid_a+c),(mid_b-d):(d-mid_b+d)]

binary_charpic = cv2.threshold(charpic,10000,255,cv2.THRESH_BINARY)[1]

#cv2.imshow('213',binary_charpic)
#
#cv2.waitKey(0)

#[10 12 13 14 14 14 15 16 16 16 17 18 20]
#[10 17 15 12 13 14 12 10 11 12  9  7 14]

[max_x,max_y] = np.where(binary_charpic == 255)

r_angle = np.arctan(4/10)

cols = img.shape[0]

rows = img.shape[1]

M = cv2.getRotationMatrix2D((int(cols/2),int(rows/2)), 360-r_angle*180/3.1415926, 1)
dst = cv2.warpAffine(img, M, (cols+10, rows+10))
cv2.imshow('rotation', dst)

cv2.waitKey(0)
print(max_x)
print(max_y)
print(r_angle)



#cv2.normalize(shift2center,0, 255, cv2.NORM_MINMAX)
#bwshift = cv2.threshold(shift2center,15,1,cv2.THRESH_BINARY_INV)[1]





#Yflag2 = Yflag.insert(1,0)




# binary_img_veri = cv2.threshold(grey_img_veri,50,255,cv2.THRESH_BINARY)[1]
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#
# bw_img = cv2.erode(binary_img,kernel,iterations=1)
#
# bw_img = cv2.erode(bw_img,kernel,iterations=1)


# print cv2.minAreaRect(bw_img)

# image, contours, hierarchy = cv2.findContours(binary_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
# cnt = contours[0]
#
# cv2.drawContours(img, cnt, 0, (0, 0, 255), 2)
#
# cv2.imshow('1',img)
#
# cv2.waitKey(0)



