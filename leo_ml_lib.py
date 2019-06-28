# Basado en el código de Leonel Aguilar: https://github.com/leaguilar/election_count_helper 
import cv2
import json
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib as mlp
import matplotlib.patches

np.set_printoptions(precision=2)

os.chdir("./handwritten_digit_recognition/")
# from wide_resnet_28_10 import WideResNet28_10
from mobilenet import MobileNet
from utils import load_mnist
os.chdir("../")

PATH = './handwritten_digit_recognition/models/'
#model_name = "WideResNet28_10"
#model=WideResNet28_10()
model_name = "MobileNet"
model=MobileNet()
model.compile()

model2=MobileNet()
model2.compile()

print('Loading pretrained weights for ', model_name, '...', sep='')
model.load_weights(PATH + model_name + "_ajustado_1" + '.h5')
model2.load_weights(PATH + model_name  + '.h5')


def mesaImagen(mesa, boleta = 1):
    fname='actas/{0:06d}'.format(mesa*10+boleta)+'.jpg'
    data_name="mesas_rv/"+'{}'.format(mesa)+'.json'
    path="./"
    out_path='./results/'

    exists = os.path.isfile(path+fname)
    
    if not exists:
        print("No se encontró el scan de la mesa ", mesa, ", boleta ", boleta)
        return None
    
    image = cv2.imread(path+fname, cv2.IMREAD_GRAYSCALE)

    p1=(380,720)
    p2=(600,1950)
    return image[p1[1]:p2[1], p1[0]:p2[0]]

def CleanRectangles(rects,thresh=25, thresh2=4):
    cleaned_rects=[]
    cleaned_rects_vcenter=[]
    cleaned_rects_hcenter=[]

    for rect in rects:
        # Draw the rectangles
        if (rect[3]>thresh) and (rect[2]>thresh2):
            cleaned_rects.append(rect)
            cleaned_rects_vcenter.append(rect[1] + rect[3]//2)
            cleaned_rects_hcenter.append(rect[0] + rect[2]//2)
    cleaned_rects_vcenter=np.array(cleaned_rects_vcenter).reshape(-1,1)
    cleaned_rects_hcenter=np.array(cleaned_rects_hcenter).reshape(-1,1)
    return (cleaned_rects,cleaned_rects_vcenter,cleaned_rects_hcenter)

def extractNumbers(image, model = model):
    # Threshold the image
    ret, im_th = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)
    im2, ctrs, hier = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    (cleaned_rects,cleaned_rects_vcenter,cleaned_rects_hcenter) = CleanRectangles(rects)

    im2, ctrs, hie = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    numeros = []
    for i,rect in enumerate(cleaned_rects):
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.2)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        if pt1 < 0:
            pt1=0
        if pt2 < 0:
            pt2=0
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

        #Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        roi = cv2.erode(roi, (6, 6))
        roi = roi / 255

        ext_digit = roi.reshape(1,28,28,1)
        prediction= model.predict(ext_digit, verbose = 0)
        val = np.argmax(prediction[0])

        numeros.append((pt1,pt2,val,prediction[0][val],roi))
    # Encontrar los renglones:
    row = 0
    rows = [[]]
    sortedDigits = sorted(numeros, key = lambda x: x[0])
    prevy = sortedDigits[0][0]
    for n in sortedDigits:
        if np.abs(n[0]-prevy) > 20:
            rows.append([n])
            row +=1
        else:
            rows[row].append(n)
        prevy = n[0]
    for i in range(0,len(rows)):
        rows[i] = sorted(rows[i], key=lambda x: x[1])

    conteos = [int("".join([str(d[2]) for d in r])) for r in rows ]
    
    return numeros, conteos

def plotDigits(ds):
    mlp.rcParams["figure.figsize"] = (20,10)
    for i in range(0,len(ds)):
        plt.subplot(5, np.ceil(len(ds)/5), i+1)
        plt.imshow(ds[i][4])
        plt.axis("off")
        plt.title("{} ( {}% )\n[{}]".format(ds[i][2], np.round(ds[i][3]*100), i ))
    plt.tight_layout()
