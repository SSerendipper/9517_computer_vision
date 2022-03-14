

"""
Package Specification:
1. Tensorflow 2.1.0
2. PyTorch 0.4.1.post2
"""

"""
Before running this program, please change the settings in the param.py.
This program is supposed to be run on an environment configured with GPU.
The command of execution is: python3 main.py
WARNING: The datasets and checkpoints of DeepWater models are not included here.
Please go to https://drive.google.com/drive/folders/1R8SF8lh6TWHVvD4ZTd6McE60kOyVZVLv?usp=sharing
to download.
This program could be run on Google Colab, please check the "Presentation.ipynb" on the Google Drive 
for more information. 
"""

from processing import Processing
from matcher import Matcher
from drawer import Drawer
from cell import Cell
from param import Params
import glob
import cv2
import matplotlib.pyplot as plt
import time

import os


def main():

    params = Params()
    try:
        path_list = os.listdir(params.path)
        path_list.sort()
        print("The files in the current folder are: ",path_list)
        data_path = params.path+str(path_list[2])
        print("The first picture is:",data_path)
        images = glob.glob(params.path + '/*.tif')
        images.sort()
        #print(images)
        print("There are",len(images),"pictures totally.")

        processing = Processing(images,params)
        matcher = Matcher(processing)
        drawer = Drawer(matcher,processing)
        masks = processing.get_masks()

        print('Generating all frames and cell states...')
        # Based on the contours for all images, tracking trajectory and mitosis image by image
        drawer.load()
        print('Successfully loaded all images')

        gen_path = params.path + "gen"
        print("Now the gen path is ",gen_path)
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
            print("gen path created successfully")

        counter = 1
        for g in drawer.get_gen_images():
            cv2.imwrite(f'{gen_path}/{counter}.tif', g)
            cv2.imwrite(f'{gen_path}/mask_{counter}.tif', masks[counter - 1])
            counter += 1
        print('Saved all images incuding processed pictures and matched pictures.')

        # Now standby for user to issue commands for retrieval
        # input: num_1 +num_2
        # 1st num represents the image number
        # 2nd num represents the cell number in the given image
        # 2nd num is not compulsory, with 1 number it will just show the image
        # press ENTER without any input will end the program
        # all inputs are assumed right

        analysis_path = params.path + "analysis"
        if not os.path.exists(analysis_path):
          os.makedirs(analysis_path)
          print("analysis path created successfully")


        while True:
            frame = None 
            cell_id = None
            string = input('Which image and cell do you want to check?(input:img NO.+cell_ID)\n')
            if len(string)>0:
                string = string.split('+')
                img_f = int(string[0])
                if len(string) > 1:
                        cell_id = int(string[1])
                        display_image = drawer.serve(img_f, cell_id)
                else:
                    display_image = drawer.serve(img_f)

                file_name = analysis_path+"/image_"+str(img_f)+"_cellID_"+str(cell_id)+".tif"
                #method plt not clear enough
                #plt.imshow(display_image)
                #plt.axis("off")
                #plt.savefig(file_name)
                cv2.imwrite(file_name,display_image)

    except ValueError:
        print("Please write a integer.")
        display_image = drawer.serve(img_f)
        



        
    
  

if __name__ == '__main__':
    main()