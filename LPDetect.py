import tkinter as tk
from tkinter import filedialog as fd
import os
import pandas as pd 
import urllib.request
import cv2 as cv
import imutils
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# PYTesseract to be installed
# pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\\tesseract.exe"

def callback():
    inputFile =  fd.askopenfilename(initialdir = "/",title = "Select file", filetypes = (("Images","*.jpg *.png *.jpeg"),("CSV","*.csv")))
    
    label1 = tk.Label(m, text=inputFile)
    label1.configure(bg="white", font=("Helvetica",10,"italic"))
    label1.place(relx = 0.5, rely = 0.4, anchor='center')

    getInput(inputFile)

    OpenCSV = tk.Button(m, text = 'Open CSV File', command=openCSV)
    OpenCSV.configure(fg="white", bg="black", font=("Helvetica",10,'bold'), width = 20, bd=1)
    OpenCSV.place(relx = 0.5, rely = 0.6, anchor = 'center') 

def getInput(inputFile):
    
    lpList = []
    fileName = 1

    if inputFile.endswith(".csv"):
        data = pd.read_csv(os.path.normpath(inputFile))
            
        
        
        for url in data['content']:
            urllib.request.urlretrieve(url,f"input\\{fileName}.jpeg")
            orgImg = cv.imread(f"input\\{fileName}.jpeg")
            licencePLateDetails = DetectionRecognition(orgImg)
            lpList.append(licencePLateDetails)
            fileName +=1
    else:
        orgImg = cv.imread(os.path.normpath(inputFile))
        cv.imwrite(f"input\\{fileName}.jpeg", orgImg)
        licencePLateDetails = DetectionRecognition(orgImg)
        lpList.append(licencePLateDetails)
    
    dict = {"licencePlate": lpList}
    df = pd.DataFrame(dict)
    df.to_csv("output\\LicencePlate.csv")


def openCSV():
    os.startfile("output\\LicencePlate.csv")


def DetectionRecognition(orgImg):

    pbtxt = "detectPlate\\graph.pbtxt"
    model = "detectPlate\\frozen_inference_graph.pb"

    cvNet = cv.dnn.readNetFromTensorflow(model,pbtxt)

    rows = orgImg.shape[0]
    cols = orgImg.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(orgImg, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.5:
            left = detection[3] * cols + 2
            top = detection[4] * rows + 2
            right = detection[5] * cols + 2
            bottom = detection[6] * rows + 2

            y1 = int(top)
            x1 = int(left)
            y2  = int(bottom)
            x2 = int(right)

            cropImg = orgImg[y1:y2,x1:x2]

            dupPlate = imutils.resize(cropImg,width=500)
            dupPlate = cv.cvtColor(dupPlate, cv.COLOR_BGR2GRAY)
            dupPlate = cv.threshold(dupPlate, 0, 250,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            dupPlate = cv.medianBlur(dupPlate, 3)
            text = pytesseract.image_to_string(Image.fromarray(dupPlate))
            return text
    




m = tk.Tk() 
m.configure(background='white')
m.title('License Plate Detector') 
m.geometry("400x400")

 
lb = tk.Label(m, text = 'Licence Plate Detector')
lb.configure(font=("Helvetica",14), background="white")
lb.place(relx = 0.5, rely = 0.1, anchor='center')

up = tk.Label(m, text='Choose an image or .csv file')
up.configure(background="white", font=("Helvetica",10))
up.place(relx = 0.4, rely = 0.3, anchor='center')

uploadButton = tk.Button(m, text = 'CHOOSE', command = callback)
uploadButton.configure(fg="white", bg="black", font=("Helvetica",10,'bold'), width = 10, bd=1)
uploadButton.place(relx = 0.8, rely = 0.3, anchor = 'center') 


m.mainloop() 