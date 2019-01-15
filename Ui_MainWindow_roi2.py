# 2018/1/4
#因應roi很多時候會包含噴頭本身 故需重新訂定roi的演算法
# real

import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtGui import *
import threading
import multiprocessing
import time
import socket
import io
import struct
from PIL import Image
import numpy as np
import queue

from UI_design.gcode_trasn import Ui_Form as Ui_Form_gcodeTrans
from UI_design.test import Ui_MainWindow
from UI_design.close_dialog import Ui_Dialog as Ui_Dialog_close
import ROI_process as roi


# 開一線程接收frame
class FrmaeGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self):

        self.stopped = False
        self.frame = []

    def start(self):

        threading.Thread(target=self.get, args=()).start()

    def get(self):
        while (not self.stopped):
            image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
            # print(image_len)
            if not image_len:
                break

            image_stream = io.BytesIO()
            image_stream.write(connection.read(image_len))
            # from PIL to opencv
            pil_image = Image.open(image_stream).convert('RGB')
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            self.frame = open_cv_image[:, :, ::-1].copy()  # Need Process
            image_stream.seek(0)

    def stop(self):
        self.stopped = True

    def getFrame(self):
        return self.frame


# gcode視窗
class gcodeWindow(QtWidgets.QWidget, Ui_Form_gcodeTrans):

    def __init__(self):
        super(gcodeWindow, self).__init__()
        self.setupUi(self)

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


# 主視窗
class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)

        # 點擊影像串流後以多線程方式無限迴圈刷新圖片以及顯示
        # self.thread_showImage = threading.Thread(target=self.showImage_fuction)  # 定义线程
        # self.show_bottom.clicked.connect(self.threading_showImage)   #以多線程 跑shownig image
        self.show_bottom.clicked.connect(self.showImage_fuction)

        # 點擊關閉後跳出關閉視窗
        self.close_botton.clicked.connect(self.click_closeButton)

        # 點擊開啟gcode傳輸視窗
        # 將原本的顯示的線程丟給gcode任務以避免gui的串流卡卡的!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 用pyqt的Qthread
        # self.gcode_button.clicked.connect(self.click_showGcodeWin)

    def click_closeButton(self):

        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog_close()
        ui.setupUi(Dialog)
        Dialog.show()
        # Dialog.exec_()

        # 關閉連線
        '''
        conn.close()     # for one photo
        socket_1.close()
        print('server close')
        '''
        '''
        connection.close()
        server_socket.close()
        print('server close')
        '''

        rsp = Dialog.exec_()
        if rsp == QtWidgets.QDialog.Accepted:  # ，QtWidgets.QDialog.Accepted  對話框的接收事件
            self.close()
            connection.close()
            server_socket.close()
            print('server close')
            # self.threading_showImage.exit()    #關閉影像串流線程
            getFrame.stop()  # 關閉抓幀子程序
            QtWidgets.QApplication.processEvents(0)  # 關閉刷新

        else:
            self.show()

    def showImage_fuction(self):  # for showing image
        # ------------------------------nozzle tracking's reference-------------------------------#
        # 判斷是否進到區域樣板的旗標
        flag_area = 1
        # 判斷是否為三步樣板的旗標
        flag_3step = 0

        # 定義template 的area搜索範圍
        area_Xmax = 500
        area_Xmin = 250
        area_Ymax = 510
        area_Ymin = 100

        # read the nozzle template
        nozzle = cv2.imread("nozzle_2errow2.PNG")
        nozzle_ref = nozzle.copy()

        # cal the fps
        fps = roi.CountsPerSec()
        fps.start()

        # for data collection
        n = 0
        flag_roi = 1
        # -----------------------------------------------------------------------------------#

        while True:

            # ----------------------Image Accept for one photo-----------------------------------#
            '''
            print('begin write image file "0001.jpg"')
            imgFile = open('test_photo/0001.jpg', 'wb')
            while True:
                imgData = conn.recv(512)  # 接收遠端主機傳來的數據
                if not imgData:
                    break  # 讀完檔案結束迴圈
                imgFile.write(imgData)
            imgFile.close()
            print('image save')
            '''
            # ----------------------Image Accept for continious capturing  with threading-----------------------------------#
            '''
            # Read the length of the image as a 32-bit unsigned int. If the
            # length is zero, quit the loop
            image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
            if not image_len:
                break

            #把下面丟到有線程的類別中
            # Construct a stream to hold the image data and read the image
            # data from the connection
            image_stream = io.BytesIO()
            image_stream.write(connection.read(image_len))

            # Rewind the stream, open it as an image with PIL and do some
            # processing on it
            image_stream.seek(0)
             # from PIL to opencv
            pil_image = Image.open(image_stream).convert('RGB')
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            img_np = open_cv_image[:, :, ::-1].copy()  # Need Process
            '''

            img_np = getFrame.getFrame()  # 線程讀取幀

            # -----------------------Image Processing-----------------------------#

            # reprocessing
            # frame_bright = np.uint8(np.clip((1.01 * img - 40), 0, 255))  # frame too high , lower the brightness
            frame_equ = roi.enhance(img_np)

            # 第一次一定先進到區域搜索的樣板內
            if flag_area == 1:
                frame_area = frame_equ[area_Ymin:area_Ymax, area_Xmin:area_Xmax]

                #cv2.imshow("d",frame_area)
                #cv2.waitKey(0)

                BOOL, img_0, center_O, nozzle_0 = roi.template_area(frame_area, frame_equ, nozzle)
                if BOOL:
                    flag_area = 0  # 成功判斷即可開始用三步搜尋法
                    flag_3step = 1  # 開啟三步樣板旗標
                    Tthreshold = 0.6
                    extend_y = 10
                    img = img_0
                    center = center_O
                    nozzle = nozzle_0



                else:
                    img = frame_equ

                # 區域搜索成功即開始使用三步樣板
            if flag_3step == 1:

                __, img, center, nozzle, Tthreshold = roi.template_3steps(frame_equ, center, nozzle, Tthreshold,
                                                                          extend_y)

                # -------------- roi   已噴頭偵測框的底邊中點   大小(24，24)---------------------
                x_leftTop = int(center[0] - 18)
                y_leftTop = int(center[1] + 25)
                x_rightBot = int(center[0] + 7)
                y_rightBot = int(center[1] + 50)

                cv2.line(img,(int(center[0] - 21),int(center[1] + 14)),(x_leftTop, y_leftTop),(0,255,0),1)
                cv2.line(img, (int(center[0] + 21), int(center[1] + 14)), (x_rightBot, y_leftTop), (0, 255, 0), 1)
                cv2.rectangle(img, (x_leftTop, y_leftTop), (x_rightBot, y_rightBot), (255, 255, 255), 1)

                img_roi = img[y_leftTop + 1: y_rightBot, x_leftTop + 1: x_rightBot]  # +1主要是去藍色的邊框
                img_roi = cv2.resize(img_roi, (60, 60))  # for showing

                '''
                # ---------------------roi data 產生------------------------------------------------------#
                # 若與前一張roi相比相差太多則不產生data
                time.sleep(5)  # 每五秒拍次~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if flag_roi == 1:  # 紀錄第一張
                    roi_ref = img_roi.copy()
                    flag_roi = 0
                    img_roi = cv2.resize(img_roi, (60, 60))  # for showing


                else:  # 之後就可以一值跟上一張比較
                    # img_roi = cv2.resize(img_roi, (24, 24))
                    # print(np.shape(img_roi))
                    # print(np.shape(roi_ref))
                    difference = cv2.subtract(img_roi, roi_ref)
                    result = not np.any(difference)  # if difference is all zeros it will return False

                    roi_ref = img_roi
                    if result is False:  # 兩張圖片不一樣
                        print(n)
                        filename = "Data_success_gold/train_{:.0f}.jpg".format(n)
                        cv2.imwrite(filename, img_roi)
                        n += 1
                        img_roi = cv2.resize(img_roi, (60, 60))  # for showing

                    else:  # 兩張圖片一樣

                        img_roi = cv2.resize(img_roi, (60, 60))  # for showing

                '''


                # -------------跟原樣板比較------------------------------#

                Hamming_distance = roi.classify_pHash(nozzle_ref, nozzle)

                # print(Hamming_distance)
                #cv2.imshow("temp", nozzle)
                if Hamming_distance >= 20:  # 若兩張噴頭的漢明距離相距過大則template更新為原本
                    nozzle = nozzle_ref
                    Tthreshold = 0.4  # 降低搜索限制
                    extend_y = 20  # 擴大區域搜索範圍
                    print("refresh the temp")

                else:
                    extend_y = 10  # 減少區域搜索範圍
                    Tthreshold = 0.6  # 提高搜索限制
                # ------------------------------------------------------#

            else:
                img = frame_equ
                img_roi = cv2.resize(nozzle, (60, 60))  # 還沒偵測到的roi就先以噴嘴代替

            #  process for showing on the interface
            image_height, image_width, image_depth = img.shape

            # img_roi = cv2.resize(nozzle, (60, 60))
            roi_height, roi_width, roi_depth = img_roi.shape

            # -------------- Qimge to show on the interface-----------------------#
            # main
            QImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            QImg = QImage(QImg.data, image_width, image_height,
                          image_width * image_depth,
                          QImage.Format_RGB888)
            self.img_main.setPixmap(QPixmap.fromImage(QImg))

            # roi
            QImg_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
            QImg_roi = QImage(QImg_roi.data, roi_width, roi_height,
                              roi_width * roi_depth,
                              QImage.Format_RGB888)
            self.img_roi.setPixmap(QPixmap.fromImage(QImg_roi))

            # -------------------------refresh the interface-------------------------------#
            QtWidgets.QApplication.processEvents()

            # ----------------------print the average -------------------------------------#
            #fps.increment()
            #print("{:.0f} photos , FPS : {:.0f} ".format(fps._num_occurrences,fps.countsPerSec()))

    def threading_showImage(self):
        # thread = threading.Thread(target=self.showImage_fuction)  # 定义线程
        # self.thread_showImage.start()  # 让线程开始工作
        pass


# show gcode transport window


if __name__ == '__main__':
    # ----------------------建立連線 for one picture------------------- #
    '''
    host = '192.168.0.108'  # 對server端為主機位置
    port = 7654
    address = (host, port)
    socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  ## AF_INET:默認IPv4, SOCK_STREAM:TCP

    socket_1.bind(address)
    socket_1.listen(1)  # 系統可以掛起的最大連接數量。該值至少為1
    print('socket startup')

    conn, addr = socket_1.accept()  # 接受遠程計算機的連接請求，建立起與客戶機之間的通信連接
    # conn是新的套接字對象，可以用來接收和發送數據。address是連接客戶端的地址
    print('Connected by', addr)
    '''

    # ----------------------建立連線 for stream------------------- #

    host = '192.168.0.108'  # 對server端為主機位置
    port = 7654
    address = (host, port)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  ## AF_INET:默認IPv4, SOCK_STREAM:TCP
    server_socket.bind(address)
    server_socket.listen(1)
    print('socket startup')

    # Accept a single connection and make a file-like object out of it
    connection = server_socket.accept()[0].makefile('rb')

    # get the frame by thread  一開始啟動子程序

    getFrame = FrmaeGet()
    getFrame.start()
    time.sleep(1)

    # --------------------------- G U I ---------------------#
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()

    # 顯示gocde視窗
    gwindow = gcodeWindow()
    window.gcode_button.clicked.connect(gwindow.handle_click)  # 連結

    window.show()
    sys.exit(app.exec_())