#copy the file nozzle_track_color.py

from __future__ import division  #python的精確除法
import cv2
import numpy as np
import time
from threading import Thread
import io
import struct
from PIL import Image



def template_area(area,img,temp):
    # 定義template 的area搜索範圍
    area_Xmax = 500
    area_Xmin = 250
    area_Ymax = 520
    area_Ymin = 100

    h, w = temp.shape[:2]  ## rows->h, cols->w
    res = cv2.matchTemplate(area, temp, 5)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


    if max_val >= 0.6:     #匹配度最大為1最小為-1

      print("area search success")
      #框出位置
      top_left = (area_Xmin + max_loc[0],area_Ymin + max_loc[1])
      bottom_right = (top_left[0] + w, top_left[1] + h)
      #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
      cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

      #計算中心點
      center = (top_left[0] + w/2, top_left[1] + h/2)

      #更新樣板
      nozzle = img[top_left[1]:top_left[1] + h , top_left[0] :top_left[0] + w]
      #nozzle = cv2.cvtColor(nozzle, cv2.COLOR_BGR2GRAY)   #轉灰階


      return (True , img , center , nozzle)

    else :
      #print("area searching")
      return (False , 0 , 0 , 0)




def template_3steps(img,center,temp,Tthreshold,extend_y):
    #先以極小區域搜索來做
    #定義搜索區域 長寬皆像外擴10pixels
    extend_x = 100
    h, w = temp.shape[:2]  ## rows->h, cols->w
    # x搜索範圍大是因為fps不穩固 只好加大搜索範圍
    Xmin = 80   #int(center[0] - w/2 - extend_x)
    Xmax = 700  #int(center[0] + w/2 + extend_x)
    Ymin = int(center[1] - h/2 - extend_y)
    Ymax = int(center[1] + h/2 + extend_y)
    area = img.copy()
    area = area[Ymin : Ymax , Xmin : Xmax]
    cv2.imshow("d", area)
    #cv2.waitKey(0)
    #print(np.shape(area))
    res = cv2.matchTemplate(area, temp, 5)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #print(max_val)

    if max_val <= Tthreshold:

        print('jump')
        Tthreshold = 0.5    #降低搜索條件
        return (False, img ,center, temp,Tthreshold)   #相差太多 反為原本不做處理

    else :

        # 框出位置
        top_left = (Xmin + max_loc[0], Ymin + max_loc[1])
        bottom_right = (top_left[0] + w, top_left[1] + h)
       # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

        # 計算中心點
        center = (top_left[0] + w / 2, top_left[1] + h / 2)

        # 更新樣板
        nozzle = img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

        # 提高搜索限制
        Tthreshold = 0.6
        '''
        # feature for ORb

        nozzle_gray = cv2.cvtColor(nozzle, cv2.COLOR_BGR2GRAY)  # 轉灰階
        nozzle_sobel = sobel(nozzle_gray)
        #nozzle_blur = cv2.GaussianBlur(nozzle_sobel, (3, 3), 0)
        #__, nozzle_sobel = cv2.threshold(nozzle_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        h2, w2 = nozzle_gray.shape[:2]  ## rows->h, cols->w
        # 樣板在img的起始點
        nozzle_origin_y = top_left[1]
        nozzle_origin_x = top_left[0]

        orb = cv2.ORB_create(edgeThreshold=10, patchSize=11, nlevels=4, fastThreshold=5, scaleFactor=1.1, WTA_K=2,
                             scoreType=cv2.ORB_FAST_SCORE, firstLevel=0, nfeatures=100)
       # orb = cv2.ORB_create(20)

        # find the keypoints with ORB
        keypoints, descriptors = orb.detectAndCompute(nozzle_sobel, None)
        nozzle_fps = cv2.drawKeypoints(nozzle_sobel, keypoints, nozzle_sobel,color=(0, 255, 0), flags=0)
        cv2.imshow("fps",nozzle_fps)
        '''

        return (True, img, center, nozzle,Tthreshold)





def hisEqulColor(img):          #CONTRAST colorful  (histogram equ)      useless
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def enhance(img):
    #sharpen
    img = cv2.GaussianBlur(img, (3, 3), 0)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, kernel)
    #gaussianblur
    #blur = cv2.GaussianBlur(sharpen,(3,3),0)
    blur = sharpen

    return blur


def sobel(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

    return dst



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#計算平均幀數
class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0    # while的蝶代數

    def start(self):
        self._start_time = time.time()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (time.time() - self._start_time)
        return     self._num_occurrences / elapsed_time  #回傳平均每秒跑多少次while





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#計算相似性


#  計算漢明距離
def Hamming_distance(hash1,hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num

# 輸入灰階圖，返回hash(似指紋的概念)
def getHash(image):
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash



def classify_pHash(image1,image2):
    image1 = cv2.resize(image1,(32,32))
    image2 = cv2.resize(image2,(32,32))
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    # 將灰階圖轉為浮點的型態來計算DCT
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    # 取左上角的8*8，这些代表图片的最低频率

    dct1_roi = dct1[0:8,0:8]
    dct2_roi = dct2[0:8,0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1,hash2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


