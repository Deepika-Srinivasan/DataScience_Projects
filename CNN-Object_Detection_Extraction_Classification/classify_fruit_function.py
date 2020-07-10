#Classification based on rotten =1 and fresh =0,model is updated to our final model
import warnings
warnings.filterwarnings("ignore")
from os import environ, chdir
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Import necessary libraries
import numpy
import matplotlib.pyplot as plt
import time
import os
import argparse

from keras.preprocessing import image
from matplotlib.image import imread

#loading the saved model 
from keras.models import load_model 
model = load_model('Fruit_Status_97_92.h5')

def classify(image_name_from_detect):
    #this part of code is going to show image name and fresh/rotten status
    
    execution_path = os.getcwd()
    output_image_path=os.path.join(execution_path,"output_image")

    #expecting user input on image name only
    #image_name="apple_group_image.jpg" 
    image_name=image_name_from_detect

    #derive extracted image folder name 
    extracted_image_folder_name=image_name + "_extract.jpg-objects" + "\\"

    #Obtain the path for extracted image folder name
    extracted_image_path=os.path.join(output_image_path,extracted_image_folder_name) #working
    
    images=[]
    fresh_fruits=[]
    rotten_fruits=[]
    i=0

    for file in os.listdir(extracted_image_path):#check how many images in that folder and loop accordingly
        filename=extracted_image_path + 'apple-' + str(i+1) + '.jpg'
        test_image1 =image.load_img(filename,target_size =(64,64))
        
        I = imread(filename)
        images.append(I)

        test_image = image.img_to_array(test_image1)
        test_image = numpy.expand_dims(test_image, axis = 0)
        result = model.predict(test_image) 
        print(result)

        if result[0][0]==1: 
            print("Filename: {}".format(file,filename))
            print("File Location: {}".format(filename))
            print("Rotten Fruit") 
            print("\n**********************\n")
            rotten_fruits.append(I)
        else: 
            print("Filename: {}".format(file,filename))
            print("File Location: {}".format(filename))
            print("Fresh Fruit") 
            print("\n**********************\n")
            fresh_fruits.append(I)
        i+=1
    
