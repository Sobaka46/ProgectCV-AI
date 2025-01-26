import cv2
import numpy as np



#-----------------------------------
#изображение
#
#
# img = cv2.imread('imgs/oleg.jpg')
#
#
# img = cv2.GaussianBlur(img, (9, 9), 0)
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.Canny(img, 100, 100)
#
# kernel = np.ones((5, 5), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)
#
# img = cv2.erode(img, kernel, iterations=1)
#
# cv2.imshow('oleg', img)
#
# cv2.waitKey(0)
#-------------------------------------


#--------------------------------------------------------
#видео с вебки

# cap = cv2.VideoCapture("video/2024-11-22 12-31-51 — копия.mkv")
#
# # cap.set(3, 500)
# # cap.set(4, 300)
#
# while True:
#     succes, img = cap.read()
#
#
#     img = cv2.GaussianBlur(img, (9, 9), 0)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.Canny(img, 30, 30)
#
#
#     kernel = np.ones((5, 5), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#
#     img = cv2.erode(img, kernel, iterations=1)
#
#     cv2.imshow('oleg', img)
#
#
#
#
#
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#---------------------------------------------------------

#-------------------------------------------------------
#создание фигур
# phto = np.zeros((450, 450, 3), dtype='uint8')
#
# # phto[100:150, 200:250] = 0, 0, 255
# cv2.rectangle(phto, (50, 50), (100, 100), (0, 0, 255), thickness=5)
#
# cv2.line(phto, (0, phto.shape[0] // 2), (phto.shape[1], phto.shape[0] // 2), (0, 0, 255), thickness=4)
#
# cv2.circle(phto, (phto.shape[1] // 2, phto.shape[0] // 2), 100, (0, 0, 255), thickness=5)
#
# cv2.putText(phto, 'oleg', (100, 150), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 255, 0), 2)
#
# cv2.imshow('Phto', phto)
# cv2.waitKey(0)
#--------------------------------------------------------

#---------------------------------------------------
#вращение
# img = cv2.flip(img, -1)

# def rotate(imgp, angle):
#     height, width = imgp.shape[:2]
#     point = (width // 2, height // 2)
#
#     mat = cv2.getRotationMatrix2D(point, angle, 1)
#     return cv2.warpAffine(imgp, mat, (width, height))
#
# # img = rotate(img, 90)
#
# def transform(imgp, x, y):
#     mat = np.float32([[1, 0, x], [0, 1, y]])
#     return cv2.warpAffine(imgp, mat, (imgp.shape[1], imgp.shape[0]))
#
# img = transform(img, 30, 100)
#-------------------------------------------------------------

#контуры
# img = cv2.imread("imgs/oleg.jpg")
#
# newimg =np.zeros(img.shape, dtype='uint8')
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (5, 5), 0)
#
# img = cv2.Canny(img, 100, 140)
#
# con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#
# cv2.drawContours(newimg, con, -1, (205, 137, 163), 1)
#
# cv2.imshow("oleg", newimg)
# cv2.waitKey(0)
#--------------------------------------------------------------------

#цветовые форматы
# img = cv2.imread("imgs/oleg.jpg")
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#
# img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# r, g, d = cv2.split(img)
#
# img = cv2.merge([r, g, d])
#
# cv2.imshow("hi", img)
# cv2.waitKey(0)

#маски
# phto = cv2.imread("imgs/oleg.jpg")
# img = np.zeros(phto.shape[:2], dtype='uint8')
#
# circle = cv2.circle(img.copy(), (200, 300), 150, 255, -1)
# square = cv2.rectangle(img.copy(), (25, 25), (250, 350), 255, -1)
#
# img = cv2.bitwise_and(phto, phto, mask=circle)
# # img = cv2.bitwise_or(circle, square)
# # img = cv2.bitwise_not(circle)
#
# cv2.imshow("hi", img)
# cv2.waitKey(0)


#распознавание лиц по фото
# img = cv2.imread('imgs/oleg.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# fases = cv2.CascadeClassifier('fases.xml')
#
# results = fases.detectMultiScale(gray, scaleFactor=7, minNeighbors=2)
#
# for (x, y, w, h) in results:
#     cv2.rectangle(img, (x, y), (x +w, y + h), (0, 0, 255), thickness=4)
#
#
# cv2.imshow("hi", img)
# cv2.waitKey(0)


# распознавание лиц по видео в ряльном времени


# функция определения лиц
def highlightFace(net, frame, conf_threshold=0.7):
    # делаем копию текущего кадра
    frameOpencvDnn=frame.copy()
    # высота и ширина кадра
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    # преобразуем картинку в двоичный пиксельный объект
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    # устанавливаем этот объект как входной параметр для нейросети
    net.setInput(blob)
    # выполняем прямой проход для распознавания лиц
    detections=net.forward()
    # переменная для рамок вокруг лица
    faceBoxes=[]
    # перебираем все блоки после распознавания
    for i in range(detections.shape[2]):
        # получаем результат вычислений для очередного элемента
        confidence=detections[0,0,i,2]
        # если результат превышает порог срабатывания — это лицо
        if confidence>conf_threshold:
            # формируем координаты рамки
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            # добавляем их в общую переменную
            faceBoxes.append([x1,y1,x2,y2])
            # рисуем рамку на кадре
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    # возвращаем кадр с рамками
    return frameOpencvDnn,faceBoxes

# загружаем веса для распознавания лиц
faceProto="opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel="opencv_face_detector_uint8.pb"

# запускаем нейросеть по распознаванию лиц
faceNet=cv2.dnn.readNet(faceModel,faceProto)

# получаем видео с камеры
video=cv2.VideoCapture(0)
# пока не нажата любая клавиша — выполняем цикл
while cv2.waitKey(1)<0:
    # получаем очередной кадр с камеры
    hasFrame,frame=video.read()
    # если кадра нет
    if not hasFrame:
        # останавливаемся и выходим из цикла
        cv2.waitKey()
        break
    # распознаём лица в кадре
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    # если лиц нет
    if not faceBoxes:
        # выводим в консоли, что лицо не найдено
        print("Лица не распознаны")
    # выводим картинку с камеры
    cv2.imshow("Face detection", resultImg)