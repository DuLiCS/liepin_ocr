import cv2

import numpy as np

def veri_seg(veri_img_path):


    img = cv2.imread(veri_img_path)

    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    binary_img = cv2.threshold(gray_img,50, 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    erode_img = cv2.erode(binary_img, kernel, iterations=1)

    x,y = np.where(erode_img==0)

    m = np.max(x)
    n = np.min(x)

    m0 = np.max(y)
    n0 = np.min(y)

    first_char = gray_img[n:m,n0:n0+(m0-n0)/4]

    second_char = gray_img[n:m,n0+(m0-n0)/4:n0+(m0-n0)*2/4]

    third_char = gray_img[n:m,n0+(m0-n0)*2/4:n0+(m0-n0)*3/4]

    fourth_char = gray_img[n:m,n0+(m0-n0)*3/4:m0]

    cv2.imwrite('seg_0.png',first_char)

    cv2.imwrite('seg_1.png',second_char)

    cv2.imwrite('seg_2.png',third_char)

    cv2.imwrite('seg_3.png',fourth_char)














