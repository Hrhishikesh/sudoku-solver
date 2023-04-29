import cv2 
import numpy as np
from utlis import *
import solver
import solver2
###### Setting image location and parameters

pathImage = "Resources/x.png"
heightImg = 450
widthImg = 450

model = initialize_model()

###### Preparing the Image


img = cv2.imread(pathImage)
img = cv2.resize(img,(widthImg,heightImg)) #Resize image to make it a square
imgBlank = np.zeros((heightImg,widthImg,3), np.uint8) #Create a blank image 
imgThreshold = preProcess(img)


###### Finding all the Contours


imgContours = img.copy() # Copy image for display purposes
imgBigContours = img.copy() # Copy image for display purposes
# External method used because we need outer contours
# Also we're using simple chain approximation
contours, hierarchy = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Find contours
cv2.drawContours(imgContours, contours , -1 , (0,255,0),3) # Draw all Detected contours


###### Finding Biggest Contour and using it as a Sudoku


biggest, maxArea = biggestContour(contours) #Find the biggest contour
if biggest.size != 0:
    biggest = reorder(biggest) # Reorder the biggest contours for using warp persepective
    cv2.drawContours(imgBigContours,biggest,-1,(0,255,0),10) #Draw biggest contour
    pts1 = np.float32(biggest) # Prepare points for warp
    pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]]) # Prepare points for warp
    matrix = cv2.getPerspectiveTransform(pts1,pts2) #Transformation matrix for warp transform
    imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))
    imgDetectedDigits = imgBlank.copy()
    #imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)


##### Split and Predict

boxes = splitBoxes(imgWarpColored)
#save_splitBoxes(imgWarpColored)
numbers = getPrediction(boxes,model)
#print(boxes[0].shape)
imgDetectedDigits = displayNumbers(imgDetectedDigits,numbers,color=(255,0,255))
numbers = np.asarray(numbers)
posArray = np.where(numbers>0,0,1)#Put 1 where we need to fill up


##### Find Solution 

board = np.array_split(numbers,9)
print(board) #Split the numbers into 9 rows
try:
    solver.solve(board)
except:
    pass
flatlist = [] #Create an empty list to convert board into simple list like before
for sublist in board:
    for item in sublist:
        flatlist.append(item)
solvedNumbers = flatlist*posArray # Get the final solution by multiplying by posArray variable from before
imgSolved = imgBlank.copy()
imgSolved = displayNumbers(imgSolved,solvedNumbers)#Display the solution
print(solvedNumbers)
print(board)

##### Overlaying the Solution
#Prepare points but this time the opposite
pts2 = np.float32(biggest)
pts1 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
#Create inverse matrix from the above two points
matrix = cv2.getPerspectiveTransform(pts1,pts2)
#Overlaying solution
imgInvWarpColored = img.copy()
imgInvWarpColored = cv2.warpPerspective(imgSolved,matrix,(widthImg,heightImg))
inv_perspective = cv2.addWeighted(imgInvWarpColored,1,img,0.5,1)#Add newly crated image and original image
#imgDetectedDigits = drawGrid(imgDetectedDigits)
#imgSolved = drawGrid(imgSolved)




## test the image
window_name = "test"
cv2.imshow(window_name,inv_perspective)
cv2.waitKey(0)
cv2.destroyAllWindows()
