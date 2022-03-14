# COMP 9517 project
# python 3.7
# cv2 3.4
import cv2
import copy
import numpy as np
from param import Params
from processing import Processing


"""
A module that is used for putting texts and drawing contours as well as trajectories
"""



# Drawer.load integrates features from Preprocessor and Matcher
# and draw contours, trajectory, path and mark mitosis
# Drawer.sever will show desired image directly

class Drawer:
    def __init__(self, matcher, processing):
        self.matcher = matcher
        self.processing = processing
        self.params = Params()
        #not-done images
        self.images_history = []
        self.gen_history = []   
        #cells in each image
        self.cells_history = []

    # Load all images and cells from Matcher
    def load(self):
        while self.processing.status!=0:
            image, cells = self.matcher.next(self.cells_history)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.images_history.append(image)
            self.cells_history.append(copy.deepcopy(cells))
            print('This is image NO.',len(self.cells_history),'and there are',len(cells),"cells in this picutre.")
        #draw the contours:
        #all the unprocessed pictures
        for i in range(len(self.images_history)):
            current_image = self.images_history[i] 
            #dictionary
            cells_in_current_img = list(self.cells_history[i].values())
            cells_area = 0
            average_size = 0
            cells_distance = 0
            average_distance = 0
            mitosis_parent_count = 0
            np.random.seed(0)
            #draw each picture
            
            for r in range(len(cells_in_current_img)):
                           #print(cells_in_current_img)
                           #print(type(cells_in_current_img))
                        
                           cell = cells_in_current_img[r]
                           cells_area+=cell.area
                           cells_distance+=cell.total_dist
                           
                           #print("cell",type(cell),cell)
                           split_flag = cell.if_split()
                           if split_flag:
                            #BGR, if split, cell_color is red,ID_color is white
                                color = (0, 0, 255)
                                line_width = 6     
                                ID_color = (255, 255, 255)
                                mitosis_parent_count +=1
                           else:
                                #each ceoll has a unique color
                                color = tuple(np.random.random(size=3) * 255)
                                line_width = 2
                                #draw the ID in the center of the cell in black
                                ID_color = (0,0,0)
                                
                           current_img = cv2.drawContours(current_image, cell.get_contour(), -1, color, line_width)
                           
                           current_img =  cv2.putText(current_image, str(cell.get_id()), cell.get_centroid(), 1, 1, ID_color, 1)
                           previous_positions = cell.get_prev_positions()
                           for k in range(len(previous_positions) - 1):
                                current_image = cv2.line(current_img, previous_positions[k], previous_positions[k + 1], color, 1)
            current_image = cv2.copyMakeBorder(current_image,80,0,0,0,cv2.BORDER_CONSTANT,0)
            average_size = cells_area/len(cells_in_current_img)
            average_distance = cells_distance/len(cells_in_current_img)
            # 2-1 to 2-4
            current_image = cv2.putText(current_image, f'2-1:Cell count: {len(cells_in_current_img)}', self.params.LOC_2_1, 1, 1,(0, 255, 0), 2)
            current_image = cv2.putText(current_image, f'2-2:Average size of all cells: {average_size}', self.params.LOC_2_2, 1, 1,(0, 255, 0), 2)
            current_image = cv2.putText(current_image, f'2-3:Average displacement of all cells: {average_distance}', self.params.LOC_2_3, 1, 1,(0, 255, 0),2)
            current_image = cv2.putText(current_image, f'2-4:Number of cell dividing: {mitosis_parent_count}', self.params.LOC_2_4,1, 1,(0, 255, 0),2)
            # Save generated image
            self.gen_history.append(current_image)
                           
                           



     # Serve currently loaded image
    # id parameter allows user to input a cell ID in the terminal
    # Motion features and physical features
    def serve(self, frame, id=None):
        frame = frame -1
        image = copy.deepcopy(self.gen_history[frame])
        if id:
            cell = self.cells_history[frame][id]
            area  = cell.area
            perimeter = cell.perimeter
            intensity = cell.intensity
            #type 3
            semi =0.3*intensity+0.3*perimeter+0.3*area
            #speed = cell.get_speed()
            total_dist = cell.get_total_dist()
            net_dist = cell.get_net_dist()
            confinement_ratio = cell.get_confinement_ratio()
            print("Area = ",area)
            print("Perimeter = ",perimeter)
            print("Intensity = ",intensity)
            print("Similarity = ",semi)
            print("Total distance taveled = ",total_dist)
            print("Net distance traveled = ",net_dist)
            print("Total trajectory time = ",frame)
            print("Confinement ratio = ",confinement_ratio)
            # Add a black padding on top for putting text 
            image = cv2.copyMakeBorder(image,80,0,0,0,cv2.BORDER_CONSTANT,0)

            image = cv2.putText(image,
                                f'Cell ID ={id} Area = {area:.2f}',
                                self.params.CELL_DETAILS_LOC_1, 1, 1, (255, 0, 255), 2)
            image = cv2.putText(image,
                                f'Perimeter = {perimeter:.2f}, Intensity = {intensity:.2f}',
                                self.params.CELL_DETAILS_LOC_2, 1, 1, (255, 0, 255), 2)
            image = cv2.putText(image,
                                f'Similarity = {semi:.2f}, Total distance taveled = {total_dist:.2f}',
                                self.params.CELL_DETAILS_LOC_3, 1, 1, (255,0, 255), 2)
            image = cv2.putText(image,
                    f'Net distance traveled = {net_dist:.2f}, Confinement ratio = {confinement_ratio:.2f}',
                    self.params.CELL_DETAILS_LOC_4, 1, 1, (255,0, 255), 2)
        return image
    
    # Returns all generated images
    def get_gen_images(self):
        return self.gen_history
    
    # Returns cell history    
    def get_cell_history(self):
        return self.cells_history