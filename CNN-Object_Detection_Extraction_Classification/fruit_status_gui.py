#ignore tensorflow warnings
import warnings
warnings.filterwarnings("ignore")
from os import environ, chdir
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Import libraries
import numpy
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np 
from keras.preprocessing import image
from keras.models import load_model 
import matplotlib.pyplot as plt
from matplotlib.image import imread
import time
import os

#Load model to use in GUI
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('Fruit_Status_97_92.h5')

#initialise GUI
top=tk.Tk()
top.geometry('500x500')
top.title('Upload and check my status')

background_image=tk.PhotoImage(file='dribbble6.png',master=top)
background_label=tk.Label(top,image=background_image)
background_label.place(relwidth=1,relheight=1) #fill fully

label=Label(top,background='#008080', font=('Courier',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    print(file_path)
    filename=file_path
    test_image =image.load_img(filename,target_size =(64,64,3))
    test_image = image.img_to_array(test_image)
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print(result)
    if result[0][0]>=0.2: 
        pred="Rotten Fruit"
        label.configure(background='#dc143c',foreground='white', text=pred,font=('Courier',20,'bold'))
    else: 
        pred="Fresh Fruit" 
        label.configure(background='#7fff00',foreground='black', text=pred,font=('Courier',20,'bold'))
    print(pred)
    
    
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Apple",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#008080', foreground='white',font=('Courier',10,'bold'))
    classify_b.place(relx=0.36,rely=0.2)
    #classify_b.pack(side=BOTTOM) cannot use place as its added every time u upload new pic
    
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)#show this button only after you upload
    except:
        pass
    
#padx=10,pady=5 external padding horizontally ,vertically 
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#008080', foreground='white',font=('Courier',10,'bold'))
upload.pack(side=TOP,pady=50)

#Specifies whether the widgets should be expanded to fill any extra space in the geometry master. 
#If false (default), the widget is not expanded.
sign_image.pack(side=TOP,expand=True)
label.pack(side=TOP,expand=True)

heading = Label(top, text="Apple Status Detector",pady=10, font=('Courier',15,'bold'))
heading.configure(foreground='#364156')
heading.pack()

top.mainloop()