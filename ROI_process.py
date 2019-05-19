#copy the file nozzle_track_color.py

from __future__ import division  #python的精確除法
import cv2
import numpy as np
import time

import pandas
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


    if max_val >= 0.65:     #匹配度最大為1最小為-1

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
        #cv2.imshow("haha",temp)
        temp = cv2.imread("nozzle_2errow2.PNG") #重新refresh樣板
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
#-------------------值方圖特徵計算方程式-----------------------------#

def calAverage(hist,total_pixel):
    sum = 0
    i = 0   #gray level
    for item in hist:
        sum = sum + item*i
        i += 1

    return sum/total_pixel


def calEntropy(hist,total_pixel):
    entropy = []
    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en


def calMostNum(hist):
    i = 0  # gray level
    cup = 0
    cup_i=0
    for item in hist:
        if item[0]>=cup :
          cup = item[0]
          cup_i = i

        i += 1
    return cup_i


def calSD(hist,average):
    sum = 0
    i = 0  # gray level
    for item in hist:
        if item[0] == 0:
            i += 1
            continue

        else:
             for x in range(1,int(item[0])) :
                 sum = sum + (i-average)**2
             i += 1


    return  np.sqrt(sum/(255+1))[0]   #[0]為了從numpy的矩陣中取直


def calVariation(average,std):
    num = std/average
    return num[0]  #[0]為了從numpy的矩陣中取直


def histogramFeatures(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    average = calAverage(hist, total_pixel)
    entropy = calEntropy(hist, total_pixel)
    mostNum = calMostNum(hist)
    std = calSD(hist, average)
    val = calVariation(average, std)
    HGfeatures = [mostNum, std, val,entropy[0]]  # 1 x 3
    return HGfeatures


#-------------------灰階共生矩陣特徵計算方程式-----------------------------#
#----------------灰階共生矩陣特徵計算方程式--------------------------#
def P(i,j,d,img,theta):
    height = img.shape[0]
    width = img.shape[1]
    sum = 0
    #零度
    if theta == 0 :
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                  if x+d < width and img[y,x] == i and img[y,x+d] == j :   #跟右邊比
                      sum += 1
                  if  x-d > -1 and img[y,x] == i and img[y,x-d] == j :   #跟左邊比
                      sum += 1
        return sum


    #45度
    if theta == 45:
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                if (x - d > -1 and y + d < height) and img[y, x] == i and img[y + d, x - d] == j:  # 跟左下比
                    sum += 1
                if (y - d > -1 and x + d < width) and img[y, x] == i and img[y - d, x + d] == j:  # 跟右上比
                    sum += 1
        return sum



     # 90度
    if theta == 90:
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                if y+d < height  and img[y , x] == i and img[y + d, x ] == j:  # 跟下面比
                    sum += 1
                if y-d > -1 and img[y, x] == i and img[y - d, x ] == j:  # 跟上面比
                    sum += 1
        return sum

    # 135度
    if theta == 135:
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                if (x+d < width  and y+d < height ) and img[y, x] == i and img[y + d, x + d] == j:  # 跟右下比
                    sum += 1
                if (y-d > -1 and x-d > -1) and img[y, x] == i and img[y - d, x - d] == j:  #  跟左上比
                    sum += 1
        return sum



def normalize_glcm(img,initial,theta):   #指定d = 1
    height = img.shape[0]
    width = img.shape[1]

    if theta == 0:
        total =  2*height*(width - 1) #水平排列總次數

    if theta == 45:
        total =  2*(height - 1)*(width - 1) #右上左下排列總次數

    if theta == 90:
        total =  2*width*(height-1) #垂直排列總次數

    if theta == 135:
        total =  2*(height - 1)*(width - 1) #左上右下排列總次數


    return initial/total



def create_GLCM(img):
    #灰階0~255先分階層

    max_gray_level = 4  # 定義最大灰度級數-------------------------------------------------------------------------------------------

    height= img.shape[0]
    width = img.shape[1]
    scope = 256 / max_gray_level

    for i in range(height):  # i 從 0 到 height-1
        for j in range(width):
            img[j, i] = (int)(img[j, i] / scope)



    # 計算灰階共生矩陣，這邊採用距離為1(d=1)，0角度---------------------------------------------------------------------------
    d = 3
    theta = 0
    # 建立灰階共生矩陣
    initial_glcm = np.zeros([max_gray_level, max_gray_level])
    for i in range(max_gray_level):  # i 從 0 到 max_gray_level-1
        for j in range(max_gray_level):
            #  #(i,j)
            # initial_glcm[j,i] = P(i,j,d,img_test,135)
            initial_glcm[j, i] = P(i, j, d, img, theta)

    # 將灰階共生矩陣規範化    把count(計數)轉變為probability(機率)
    glcm = normalize_glcm(img, initial_glcm, theta)

    return  glcm


# ASM （angular second moment)特征（或称能量特征）
def cal_asm(glcm,height,width):
    sum_asm = 0.0
    for i in range(height):  # i 從 0 到 height-1
        for j in range(width):
            a = glcm[j, i] * glcm[j, i]
            sum_asm += a

    #print("asm = {:f}".format(sum_asm))
    return sum_asm


# 对比度（Contrast）
def cal_contrast(glcm,height,width):
    sum_contrast = 0.0
    for i in range(height):  # i 從 0 到 height-1
        for j in range(width):
            a = (i - j) * (i - j) * glcm[j, i]
            sum_contrast += a

    #print("contrast = {:f}".format(sum_contrast))
    return sum_contrast

# 熵（entropy）
def cal_GLCMentropy(glcm,height,width):
    sum_entropy = 0.0
    for i in range(height):  # i 從 0 到 height-1
        for j in range(width):
            if glcm[j, i] != 0:
                a = -1 * glcm[j, i] * np.log(glcm[j, i])
                sum_entropy += a

    #print("entropy = {:f}".format(sum_entropy))
    return sum_entropy

#  逆差矩（IDM：Inverse Difference Moment）
def cal_IDM(glcm,height,width):
    sum_idm = 0
    for i in range(height):  # i 從 0 到 height-1
        for j in range(width):
            sum_idm += (1 / (1 + (i - j) * (i - j))) * glcm[j, i]

    #print("idm = {:f}".format(sum_idm))
    return  sum_idm

def glcmFeatures(img):
    # 灰階共生矩陣特徵計算
    glcm = create_GLCM(img)
    height = glcm.shape[0]
    width = glcm.shape[1]

    asm = cal_asm(glcm, height, width)
    contrast = cal_contrast(glcm, height, width)
    GLCMentropy = cal_GLCMentropy(glcm, height, width)
    idm = cal_IDM(glcm, height, width)
    GLCMfeatures = [asm, contrast, GLCMentropy, idm]
    return GLCMfeatures

def getQueueValue(q,qq):
    values = []
    i = 100
    while not q.empty():
      value = q.get()
      values.append(value)
      qq.put(value)

    while not qq.empty():  #再將q補滿以免其為空
      value = qq.get()
      q.put(value)

    return values