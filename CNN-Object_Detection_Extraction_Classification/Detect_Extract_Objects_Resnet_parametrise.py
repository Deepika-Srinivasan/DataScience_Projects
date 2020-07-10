#classification is based on rotten =1 else its fresh
#Import libraries 
#Suppress tensorflow warnings
import warnings
warnings.filterwarnings("ignore")
from os import environ, chdir
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import keras
import numpy 
import scipy 
#import pillow 
import matplotlib 
import h5py 
import keras
import imageai
import os
from imageai.Detection import ObjectDetection
from IPython.display import Image

import classify_fruit_function_Version2
#import classify_fruit_function_Version3

import argparse

parser = argparse.ArgumentParser(description='Arguments obtained from user')
parser.add_argument('image_name', type=str, help='Image name')
parser.add_argument('percentage', type=str, help='Percentage to detect images')
args = parser.parse_args()

print("Image name provided is -",args.image_name)
print("Percentage provided is -",args.percentage)

#print(type(args.percentage))

#Parametrise input image path,output image path,input image name
execution_path = os.getcwd()
input_image_path=os.path.join(execution_path,"input_image")
output_image_path=os.path.join(execution_path,"output_image")
print("Input images should be in :", input_image_path)
print("Output images will be saved in :", output_image_path)

#input_image_name=args.image_name #this is the only input needed from user
#input_image_name="apple_group_image.jpg"
input_image_name=args.image_name
output_image_name=input_image_name + "_extract.jpg"

print("\n***********************************************\n")
print(os.path.join(input_image_path , args.image_name))

#we are trying to extract objects and store in a new folder name same as output file 
import warnings
warnings.filterwarnings("ignore")
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(apple=True)
detections,objects_path = detector.detectCustomObjectsFromImage(input_image=os.path.join(input_image_path , input_image_name), output_image_path=os.path.join(output_image_path , output_image_name), custom_objects=custom_objects, minimum_percentage_probability=int(args.percentage),extract_detected_objects=True)


extracted_image_folder_name= output_image_name + "-objects" + "\\"
print("Extracted images are stored in :",extracted_image_folder_name)
extracted_image_path=os.path.join(output_image_path,extracted_image_folder_name)
print("Extracted images path is :",extracted_image_path)

print(detections)

print("Number of apple detected-",len(detections))

print("\n*****************Extracted Images with Probability************************************\n")
for eachObject, eachObjectPath in zip(detections, objects_path):
    print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : " )
    #print("Object's image saved in " + eachObjectPath)
    print("--------------------------------")
    
classify_fruit_function.classify(args.image_name)
