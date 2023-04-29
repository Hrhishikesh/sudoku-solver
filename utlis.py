##### For Relative functions #####

import cv2
import numpy as np
from tensorflow.keras.models import load_model


### Preprocessing Image

def preProcess(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Convert Image to Gray Scale
    imgBlur = cv2.GaussianBlur(imgGray,(1,1),1) # Add Gaussian Blur
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,1,1,9,2) # Apply Adaptive Threshold
    return imgThreshold


### Finding the Biggest Contour

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    
    for i in contours: # Looping through all the contours one by one
        area = cv2.contourArea(i) #Check area of contour i 
        # We don't want contours to be very small, very small means noise
        if area > 50:
            par = cv2.arcLength(i, True) # Find parameter of contour
            approx = cv2.approxPolyDP(i,0.02*par,True) # Find how many corners does it have
            if area>max_area and len(approx) == 4: # len(approx) == 4 means it's either a rectangle or a square
                biggest = approx 
                max_area = area
    return biggest,max_area # biggest contains all the corner points, max_area is maximum area of biggest contour


### Reorder points for Warp Perspective

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


### Splitting the Image into different boxes

def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    
    return boxes

### Saving Image boxes for testing

def save_splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes = []
    n=0
    for r in rows:
        cols = np.hsplit(r,9)
        
        for box in cols:
            cv2.imwrite("test_imgs\{}.jpg".format(n),box)
            n=n+1
    
    print("Pictures saved")

### Load the model

def initialize_model():
    model = load_model('modelsolver_v1.h5')
    return model



### Get Prediction of Images

def getPrediction(boxes,model):
    results = []
    for image in boxes:

        ## Prepare image
        img = np.array(image)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = img[4:img.shape[0] - 4,4:img.shape[1] - 4]
        #img= img.reshape((img.shape[0], 28, 28, 1)).astype('float32')
        img = cv2.resize(img,(28,28))
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(img_gray, 155, 255, cv2.THRESH_BINARY)
        bwimg = cv2.bitwise_not(blackAndWhiteImage)
        img = bwimg.reshape(1,28,28,1).astype('float32')
        img = img/255
        #img = img.reshape(1,28,28,1)

        ## Get Prediction
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.amax(predictions)
        print(classIndex,probabilityValue)

        ## Save in results list
        if probabilityValue > 0.8:
            results.append(classIndex)
        else:
            results.append(0)

    return results


### Display the Numbers

def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0,9):
        for y in range(0,9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img,str(numbers[(y*9)+x]),
                            (x*secW+int(secW/2)-10, int((y+0.8)*secH)),cv2.FONT_HERSHEY_COMPLEX_SMALL
                            ,2,color,2,cv2.LINE_AA)
                
    return img

    
