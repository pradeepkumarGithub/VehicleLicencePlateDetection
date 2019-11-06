import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
import os, io 
import pandas as pd 
import urllib.request
import cv2 as cv
from google.cloud import vision

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"detectPlate\\GoogleCredentials.json"
client = vision.ImageAnnotatorClient()


inputFile = None
lpList = []
textList = ["No Licence Plate Detected","Unsupported file format","Unknown Exception"]

def callback():
    
    global inputFile
    inputFile =  fd.askopenfilename(initialdir = "D:",title = "Select file", 
                    filetypes = (("Image","*.jpg *.png *.jpeg"),("Video","*.mp4"),("CSV","*.csv")))
    fileName.set(inputFile)

    label1 = tk.Label(m, text="Uploaded File Path")
    label1.configure(bg="#1e1e1e", font=("Helvetica",12,"bold"), fg="white")
    label1.place(relx = 0.5, rely = 0.5, anchor='center')

    label2 = tk.Label(m, textvariable=fileName)
    label2.configure(bg="#1e1e1e", font=("Times",12,'italic'), fg="white")
    label2.place(relx = 0.5, rely = 0.55, anchor='center')
    
    detectButton = tk.Button(m, text = 'DETECT', relief=GROOVE, command= setStatus)
    detectButton.configure(fg="black", bg="#6fff00", font=("Helvetica",12,'bold'), width = 10, height=1, bd=0)
    detectButton.place(relx = 0.5, rely = 0.65, anchor = 'center')



def setStatus():
      
    getInput(inputFile)

    op = tk.Label(m, text='View the result')
    op.configure(bg="#1e1e1e", font=("Helvetica",12,"bold"), fg="white")
    op.place(relx = 0.5, rely = 0.8, anchor='center')

    openButton = tk.Button(m, text = 'Click Here', relief=GROOVE, command=openCSV)
    openButton.configure(fg="black", bg="#6fff00", font=("Helvetica",12,'bold'), width = 10, height=1, bd=0)
    openButton.place(relx = 0.5, rely = 0.87, anchor = 'center') 
    

def getInput(inputFile):
    
    fileNum = 0
    
    if inputFile.endswith(".csv"):
        data = pd.read_csv(os.path.normpath(inputFile))
            
            
        for url in data['content']:
            urllib.request.urlretrieve(url,f"input\\{fileNum}.jpeg")
            orgImg = cv.imread(f"input\\{fileNum}.jpeg")
            licencePLateDetails = DetectionRecognition(orgImg)
            print(licencePLateDetails)
            lpList.append(licencePLateDetails)
            fileNum +=1


    if inputFile.endswith(".mp4"):
        vidcap = cv.VideoCapture(inputFile)
        def getFrame(sec):
            vidcap.set(cv.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = vidcap.read()
            
            if hasFrames:
                licencePLateDetails = DetectionRecognition(image)
                
                if licencePLateDetails not in textList:
                    lpList.append(licencePLateDetails)
                    print(licencePLateDetails)
            
            return hasFrames
                

        sec = 0
        frameRate = 3
        success = getFrame(sec)
        fileNum +=1
        while success:
            sec = sec + frameRate
            sec = round(sec,2)
            success = getFrame(sec)

    
    else:
        orgImg = cv.imread(os.path.normpath(inputFile))
        cv.imwrite(f"input\\{fileNum}.jpeg", orgImg)
        licencePLateDetails = DetectionRecognition(orgImg)
        print(licencePLateDetails)
        lpList.append(licencePLateDetails)


    
    dict = {"licencePlate": lpList}
    lp = pd.DataFrame(dict)
    lp.to_csv("output\\LicencePlate.csv")


    
def liveVideo():
      
    
    cap = cv.VideoCapture(0)

    while(True):
    
        check, frame = cap.read()
        licencePLateDetails = DetectionRecognition(frame)
        if licencePLateDetails not in textList:
            lpList.append(licencePLateDetails)
            print(licencePLateDetails)
        
    
        cv.imshow("LiveCam - Press 'Q' to exit LiveCam",frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    dict = {"licencePlate": lpList}
    lp = pd.DataFrame(dict)
    lp.to_csv("output\\LicencePlate.csv")
    
       
    op = tk.Label(m, text='View the result')
    op.configure(bg="#1e1e1e", font=("Helvetica",12,"bold"), fg="white")
    op.place(relx = 0.5, rely = 0.55, anchor='center')

    openButton = tk.Button(m, text = 'Click Here', relief=GROOVE, command=openCSV)
    openButton.configure(fg="black", bg="#6fff00", font=("Helvetica",12,'bold'), width = 10, height=1, bd=0)
    openButton.place(relx = 0.5, rely = 0.65, anchor = 'center') 

    cap.release()
    cv.destroyAllWindows()



def openCSV():
    os.startfile("output\\LicencePlate.csv")
    m.destroy()


def DetectionRecognition(orgImg):

    try: 
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

                
                cv.imwrite(f"output\\temp.jpeg", cropImg)
                     
                image_path = f'output\\temp.jpeg'

                with io.open(image_path, 'rb') as image_file:
                    content = image_file.read()

                image = vision.types.Image(content=content)

                response = client.text_detection(image=image) 
                df = pd.DataFrame(columns=['locale', 'description'])

                texts = response.text_annotations
                for text in texts:
                    df = df.append(
                        dict(
                            locale=text.locale,
                            description=text.description
                        ),
                        ignore_index=True
                    )

                text = df['description'][0]
                
                os.remove('output\\temp.jpeg')
                return text

            else:
                return "No Licence Plate Detected"
    
    except AttributeError:
        return ("Unsupported file format")
    
    except Exception as e:
        return str(e)

    




m = tk.Tk() 
m.title('License Plate Detector') 
m.configure(background='#1e1e1e')
m.geometry("600x650")
m.resizable(0,0)

 
lb = tk.Label(m, text = 'Licence Plate Recognition')
lb.configure(font=("Verdana",14), background="#1e1e1e", fg="white")
lb.place(relx = 0.5, rely = 0.1, anchor='center')

up = tk.Label(m, text='Upload image, video \nor .csv file')
up.configure(background="#1e1e1e", font=("Helvetica",12), fg="white")
up.place(relx = 0.25, rely = 0.25, anchor='center')

uploadButton = tk.Button(m, text = 'UPLOAD', command = callback)
uploadButton.configure(fg="black", bg="#6fff00", font=("Helvetica",12,'bold'), width = 15, height=1, bd=0)
uploadButton.place(relx = 0.25, rely = 0.35, anchor = 'center') 

up = tk.Label(m, text='OR')
up.configure(background="#1e1e1e", font=("Helvetica",12), fg="white")
up.place(relx = 0.5, rely = 0.3, anchor='center')

live = tk.Label(m, text='Live LPR')
live.configure(background="#1e1e1e", font=("Helvetica",12), fg="white")
live.place(relx = 0.75, rely = 0.25, anchor='center')

camButton = tk.Button(m, text = 'OPEN CAMERA', command = liveVideo)
camButton.configure(fg="black", bg="#6fff00", font=("Helvetica",12,'bold'), width = 15, height=1, bd=0)
camButton.place(relx = 0.75, rely = 0.35, anchor = 'center') 

fileName = StringVar()
status = StringVar()

m.mainloop() 