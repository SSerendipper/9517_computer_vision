# COMP 9517 project
# python 3.7
# cv2 3.4
import cv2
import numpy as np
from cell import Cell
from param import Params
from processing import Processing

"""
mathcer responses for calculating, updating and analysing all information for cells
it will return all cell information in a image, for example,contour, perimeter,cent, area 
parameters below are used to match cell trajectory and mitosis
"""
DIST_THRESHOLD = 50
### detect split
MIN_SPLIT_RATIO = 0.2
MAX_SPLIT_RATIO = 0.8

MIN_SIM_THRESHOLD = 0.5
MAX_SIM_THRESHOLD = 1.5

MAX_DIS_RATIO = 2


class Matcher:

    def __init__(self, processing):
        self.processing = processing
        self.params = Params()
        self.existing_cells = {}
        # For each frame, all basic information for cells will be stored in the dict
        # with function register
        self.id_counter = 0
        self.flag = 0

    # Register a new cell
    # It will initialize basic info for a cell, including contour, perimeter,cent, area, intensity
    def register(self, contour, perimeter,cent, area, intensity):
        self.id_counter += 1
        new_cell = Cell(self.id_counter, contour, perimeter,cent, area, intensity)
        self.existing_cells[self.id_counter] = new_cell

    # Get a set of new contours, generate centroids and match new cells to existing cells.
    # The main module for matcher compares cells in the current frame and the previous frame,
    # and updates cell information if a cell has a trajectory or it has just split in the current frame.
    # "cells_history" is a parameter that belongs to drawer.py.
    # Each element in it logs information of all cells for a frame
    def next(self, cells_history):
        # Get the next picture and set of contours from Processing
        #image is already in gray sacle
        img, contours = self.processing.next()

        # Calculate centroids,areas,perimeters and intensities
        # intensity is defined as the average gray level of contour pixels
        cents, areas,perimeters,intensities = self.__get_cents_areas_perimeters__intensities(contours,img)
      

        # If there are no existing cells, add all new cells       
        if not cells_history:
            # For first frame just log all new cells' basic information
            for contour, perimeter, cent, area, intensity in zip(contours, perimeters,cents, areas, intensities):
                self.register(contour, perimeter,cent, area, intensity)
            return img, self.existing_cells
        else:
            # For other frames, initialize needed info also register new cell info in existing_cells
            pre_cells = cells_history[-1]
            self.existing_cells = {}
            self.id_counter = 0
            
            for contour, perimeter,cent, area, intensity in zip(contours,perimeters, cents, areas, intensities):
                self.register(contour, perimeter,cent, area, intensity)

            ## Perform matching and update matched cells, or add new cell if min_dist < DIST_THRESHOLD
            # key is the id of an existing cell
            # old is the key for pre_cells
            # sim represents similarity. This is used for measuring the similarity between cells
            #type 1: "only_area_min"
            #type 2: "only_intensity_min"
            #type 3: "only_area_min_sec"
            #type 4:"0.5*area+0.5*intensity_min_sec"
            #type 5: "0.5*area+0.5*intensity_min"
            #check by the number of cells
            self.flag += 1
            for old in pre_cells:
                # Find the closest 5 cells in the current frame to the cell in the previous frame
                # also get their information
                # min represents the closest cell to the old cell
                # sec represents the 2nd closest cell to the old cell
                old_cent = pre_cells[old].get_centroid()
                old_contour = pre_cells[old].get_contour()
                old_perimeter =pre_cells[old].perimeter
                old_area = pre_cells[old].get_area()
                old_intensity = pre_cells[old].intensity

                ### compare each old cell with 3 new closest cells and update new cells based on the result
                distances = [(self.__distance__(self.existing_cells[key].get_centroid(), old_cent), key) for key in
                             self.existing_cells]
                
                min_dist, min_key = sorted(distances, key=lambda x: x[0], reverse=False)[0]
                sec_dist, sec_key = sorted(distances, key=lambda x: x[0], reverse=False)[1]
                #cell_details
                min_area = self.existing_cells[min_key].get_area()
                sec_area = self.existing_cells[sec_key].get_area()
                
                #check if mitosis is now under way
                #if both closest 2 cells are spliting, assuming mitosis is now under way
                if MIN_SPLIT_RATIO <= min_area / old_area <= MAX_SPLIT_RATIO and \
                        MIN_SPLIT_RATIO <= sec_area / old_area <= MAX_SPLIT_RATIO and \
                        sec_dist < DIST_THRESHOLD:
                    # If satisfied then the 3 cells will be marked as under mitosis
                    # Their status relating to mitosis will be updated
                    pre_cells[old].split_p = True
                    self.existing_cells[min_key].split_c = True
                    self.existing_cells[sec_key].split_c = True
                    

                #cell_details
                min_intensity = self.existing_cells[min_key].intensity
                sec_intensity = self.existing_cells[sec_key].intensity
                
                min_perimeter = self.existing_cells[min_key].perimeter
                sec_perimeter = self.existing_cells[sec_key].perimeter
                
                min_cent = self.existing_cells[min_key].get_centroid()
                sec_cent = self.existing_cells[sec_key].get_centroid()
                
                min_contour = self.existing_cells[min_key].get_contour()
                sec_contour = self.existing_cells[sec_key].get_contour()
                if pre_cells[old].split_p == False:
                    if self.params.similarity_type =="only_area_min":
                    #only concern the closest cell
                    #check by the ratio of min_area / old_area
                    
                        if min_dist < DIST_THRESHOLD and MIN_SIM_THRESHOLD <= min_area / old_area <= MAX_SIM_THRESHOLD and \
                            pre_cells[old].split_p == False:
                        # if the old cell is not splitting then consider the area with the closest cell
                        # if satisfied then update the new cell's status of its trajectory
                            self.existing_cells[min_key] = pre_cells[old]
                            self.existing_cells[min_key].id = min_key
                            self.existing_cells[min_key].split_c = False
                            self.existing_cells[min_key].update(min_contour, min_perimeter,min_cent, min_area, min_intensity)
                            
                    elif self.params.similarity_type =="only_intensity_min":
                    #only concern the closest cell
                    #check by the ratio of min_intensity / old_intensity 
                        if min_dist < DIST_THRESHOLD and MIN_SIM_THRESHOLD <= min_intensity / old_intensity <= MAX_SIM_THRESHOLD and \
                            pre_cells[old].split_p == False:
                        # if the old cell is not splitting then consider the area with the closest cell
                        # if satisfied then update the new cell's status of its trajectory
                            self.existing_cells[min_key] = pre_cells[old]
                            self.existing_cells[min_key].id = min_key
                            self.existing_cells[min_key].split_c = False
                            self.existing_cells[min_key].update(min_contour, min_perimeter,min_cent, min_area, min_intensity)       
                            
                    elif self.params.similarity_type =="only_area_min_sec":
                        if sec_dist < DIST_THRESHOLD and pre_cells[old].split_p == False: 
                            min_semi = min_area / old_area
                            sec_semi = sec_area / old_area
                            if MIN_SIM_THRESHOLD <= min_semi <= MAX_SIM_THRESHOLD and MIN_SIM_THRESHOLD <= sec_semi <= MAX_SIM_THRESHOLD:
                                if abs(min_semi-1)<abs(sec_semi-1):
                                    self.existing_cells[min_key] = pre_cells[old]
                                    self.existing_cells[min_key].id = min_key
                                    self.existing_cells[min_key].split_c = False
                                    self.existing_cells[min_key].update(min_contour, min_perimeter,min_cent, min_area, min_intensity)
                                else:
                                    self.existing_cells[sec_key] = pre_cells[old]
                                    self.existing_cells[sec_key].id = min_key
                                    self.existing_cells[sec_key].split_c = False
                                    self.existing_cells[sec_key].update(sec_contour, sec_perimeter,sec_cent, sec_area, sec_intensity)
                                    
                    elif self.params.similarity_type =="0.5*area+0.5*intensity_min_sec":
                        if sec_dist < DIST_THRESHOLD and pre_cells[old].split_p == False:      
                            min_semi = 0.5*min_area / old_area+0.5*min_intensity / old_intensity
                            sec_semi = 0.5*sec_area / old_area+0.5*sec_intensity / old_intensity
                            if MIN_SIM_THRESHOLD <= min_semi <= MAX_SIM_THRESHOLD and MIN_SIM_THRESHOLD <= sec_semi <= MAX_SIM_THRESHOLD:
                                if abs(min_semi-1)<abs(sec_semi-1):
                                    self.existing_cells[min_key] = pre_cells[old]
                                    self.existing_cells[min_key].id = min_key
                                    self.existing_cells[min_key].split_c = False
                                    self.existing_cells[min_key].update(min_contour, min_perimeter,min_cent, min_area, min_intensity)
                                else:
                                    self.existing_cells[sec_key] = pre_cells[old]
                                    self.existing_cells[sec_key].id = min_key
                                    self.existing_cells[sec_key].split_c = False
                                    self.existing_cells[sec_key].update(sec_contour, sec_perimeter,sec_cent, sec_area, sec_intensity)
                    elif self.params.similarity_type =="0.5*area+0.5*intensity_min":
                        min_semi = 0.5*min_area / old_area+0.5*min_intensity / old_intensity
                        if min_dist < DIST_THRESHOLD and MIN_SIM_THRESHOLD <= min_semi <= MAX_SIM_THRESHOLD and \
                            pre_cells[old].split_p == False:
                        # if the old cell is not splitting then consider the area with the closest cell
                        # if satisfied then update the new cell's status of its trajectory
                            self.existing_cells[min_key] = pre_cells[old]
                            self.existing_cells[min_key].id = min_key
                            self.existing_cells[min_key].split_c = False
                            self.existing_cells[min_key].update(min_contour, min_perimeter,min_cent, min_area, min_intensity)   
                        

                    
                          
        return img, self.existing_cells
                    
                    
                         
                        
                            
                    
                          
                  
                
                
 


    # get the distance of 2 centroids
    # Euclidean Distance is used
    def __distance__(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # get the centroids,area,perimeters and intensity
    def __get_cents_areas_perimeters__intensities(self, contours,img):
        cents = []
        areas = []
        perimeters = []
        average_intensities = []
        for i, j in zip(contours, range(len(contours))):
            # using moments in cv2 to calculate the centroid for each cell
            M = cv2.moments(i)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #save the center position in cents
            cents.append((cX, cY))
            areas.append(cv2.contourArea(i))
            perimeters.append(cv2.arcLength(i,True))
        #len(contours) = number of cells
        
        #intensity = the intensities of cells on contour/the number of cells on contour
        for i in range(len(contours)):
            intensity = 0
            point_numbers = len(contours[i])
            target = str(contours[i].reshape(-1)).replace('\n',',').replace(' ',',').replace(',,',',').replace(',,',',').replace('[,',''.replace('],',''))
            if target[0]!='[':
                target_1=target[:-1]
           
            else:
                target_1 = target[1:-1]
            target_2 = target_1.split(",")
            #print(target,target_2)
            target = list(map(int,target_2))
            for p in range(1,int(len(target)/2),2):
                for r in range(0,int(len(target)/2),2):
                    #print(p)
                    #print(r)
                    x = target[p]
                    y = target[r]
              
                    #print(x,y)
                    intensity+=img[x][y]
            avg_intensity = intensity/point_numbers
            average_intensities.append(avg_intensity)
        return cents, areas,perimeters,average_intensities

 