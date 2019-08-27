import pandas as pd 
import cv2
import urllib.request
import imutils
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import tkinter as tk 
from tkinter import filedialog as fd
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

def callback():
    inputFile =  fd.askopenfilename(initialdir = "/",title = "Select file", filetypes = (("Images","*.jpg, *.png, *.jpeg"),("CSV","*.csv")))
    
    label1 = tk.Label(m, text=inputFile)
    label1.configure(bg="white", font=("Helvetica",10,"italic"))
    label1.place(relx = 0.5, rely = 0.4, anchor='center')

    getInput(inputFile)

    # detectButton = tk.Button(m, text = 'Detect Licence Plate', command=getInput(inputFile))
    # detectButton.configure(fg="white", bg="black", font=("Helvetica",10,'bold'), width = 20, bd=1)
    # detectButton.place(relx = 0.5, rely = 0.6, anchor = 'center') 

    # OpenCSV = tk.Button(m, text = 'Open CSV File', command=openCSV())
    # OpenCSV.configure(fg="white", bg="black", font=("Helvetica",10,'bold'), width = 20, bd=1)
    # OpenCSV.place(relx = 0.5, rely = 0.6, anchor = 'center') 

# def openCSV():
#     os.startfile("abcd.csv")

    
def getInput(inputFile):
    
    lpList = []

    if inputFile.endswith(".csv"):
        data = pd.read_csv(os.path.normpath(inputFile))
            
        fileName = 1
        
        for url in data['content']:
            urllib.request.urlretrieve(url,f"{fileName}.jpeg")
            orgImg = cv2.imread(f"{fileName}.jpeg")
            licencePLateDetails = DetectionRecognition(orgImg)
            lpList.append(licencePLateDetails)
            fileName +=1
    else:
        orgImg = cv2.imread(os.path.normpath(inputFile))
        licencePLateDetails = DetectionRecognition(orgImg)
        lpList.append(licencePLateDetails)
    
    dict = {'licencePlate': lpList}
    df = pd.DataFrame(dict)
    df.to_csv('LicencePlate.csv')

def DetectionRecognition(orgImg):
        
    orgImg = imutils.resize(orgImg,width=500)
    gsImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2GRAY)
    gsImg = cv2.bilateralFilter(gsImg, 11, 17, 17)
    gsImg = cv2.Canny(gsImg, 170, 200)

    contourCounts = 0
    contourCounts, new = cv2.findContours(gsImg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contourCounts = sorted(contourCounts, key=cv2.contourArea,reverse = True)[:30]
    NumberPlateCnt = None

    index = 0
    for index in contourCounts:
        perimeter = cv2.arcLength(index, True)
        approx = cv2.approxPolyDP(index, 0.02 * perimeter, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            x,y,w,h = cv2.boundingRect(index)
            licencePlate = orgImg[y:y + h,x:x + w]
            break

    if licencePlate is None:
            return "No Licence Plate Detected"
        
    else:
        dupPlate = imutils.resize(licencePlate,width=500)
        dupPlate = cv2.cvtColor(dupPlate, cv2.COLOR_BGR2GRAY)
        dupPlate = cv2.threshold(dupPlate, 0, 250,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        dupPlate = cv2.medianBlur(dupPlate, 3)
        

    text = pytesseract.image_to_string(Image.fromarray(dupPlate))
    if text is None:
        return "No Text is recognized"
    else:
        return text


m = tk.Tk() 
m.configure(background='white')
m.title('License Plate Detector') 
m.geometry("400x400")

 
lb = tk.Label(m, text = 'Licence Plate Detector')
lb.configure(font=("Helvetica",14), background="white")
lb.place(relx = 0.5, rely = 0.1, anchor='center')

up = tk.Label(m, text='UPLOAD AN IMAGE OR CSV FILE')
up.configure(background="white", font=("Helvetica",10))
up.place(relx = 0.4, rely = 0.3, anchor='center')

uploadButton = tk.Button(m, text = 'Click Here', command = callback)
uploadButton.configure(fg="white", bg="black", font=("Helvetica",10,'bold'), width = 10, bd=1)
uploadButton.place(relx = 0.8, rely = 0.3, anchor = 'center') 


m.mainloop() 





    


