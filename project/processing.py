import cv2
import numpy as np 
from param import Params
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
class Processing():
  def __init__(self, images,params):
    self.remaining_img = len(images)
    self.image_address = images
    self.params = Params()
    #tracking the present iamge on next call
    self.counter = 0
    #100 = not done,0 = done
    self.status = 100
    #save all the processed pictures
    self.processed = self.processed(images)
    
    #watershed
  def preprocess_1(self,img):
    kernel = np.ones((3,3),np.uint8)
    # open and close for removing bg noise
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    img = cv2.GaussianBlur(opening, (3, 3), 0)
    #img = np.array(img)
    # threshold = 129
    thresh, img = cv2.threshold(img, 129, 255, cv2.THRESH_BINARY)
    #watershed
    distance = ndi.distance_transform_edt(img)
    markers = ndi.label(peak_local_max(distance ,footprint=np.ones((3, 3)), indices=False,labels=img))[0]
    ws_labels = watershed(-distance, markers, mask=img)
    #print(ws_labels)
    return ws_labels
  #only threshold
  def preprocess_2(self,img):
    kernel=np.ones((2,2),np.uint8) #closing
    dilation=cv2.dilate(img,kernel,iterations=5) 
    erosion=cv2.erode(dilation,kernel,iterations=5) 
    ret, thresh = cv2.threshold(dilation, 150, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # threshold
    img_thresh1 = cv2.GaussianBlur(thresh,(3,3),0)
    return img_thresh1
  # Process an array of original images and return segment pictures
  def processed(self,images):
    try:
      #all images after processing
      processed = []
      #print(self.image_address)
      params = Params()
      for i in self.image_address:
        #print(i)
        img = cv2.imread(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if params.preprocess == "preprocess_1":
          img = self.preprocess_1(gray)
          processed.append(img)
        elif params.preprocess == "preprocess_2":
          img = self.preprocess_2(gray)
          processed.append(img)

      print(len(processed),"pictures have been processed.")
      plt.imshow(img)
      plt.savefig("./test.png")
      print("Checking the last picture in test.png")
      return processed
    #except ValueError:
      #print("operands could not be broadcast together with different shapes.")

    except cv2.error:
      print("There is somethin wrong,check the address of pictures again.")

        
        

  def next(self):
    # Indicate when done
    if self.counter >= self.remaining_img - 1:
        self.status = 0

    # Serves the next image and its mask
    image = cv2.imread(self.image_address[self.counter],cv2.IMREAD_GRAYSCALE)
    processed = self.processed[self.counter]
    processed = processed.astype(np.uint8)
    contours = []
    contours,_=cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours1=[]
    for i in contours:
      #area.append(cv2.contourArea(i))
      if cv2.contourArea(i)>self.params.min_cell_area: 
        contours1.append(i)

    self.counter += 1
    return image, contours1

  def get_masks(self):
        return self.processed

