import cv2
# FROM MTCNN PACKAGE IMPORT MTCNN CLASS
from mtcnn import MTCNN

# MAKING AN OBJECT OF CLASS MTCNN

detector = MTCNN() 

# USING OPENCV TO OPEN AN IMAGE
# THIS IMAGE IS CONVERTED INTO ARRAY

img = cv2.imread('images/IMG_1631.jpeg') #imread IS A FUNCTION AND GIVING IT THE PATH OF THE IMAGE 
#img = cv2.imread('images/IMG_1622.jpeg')
#img = cv2.imread('images/IMG_1630.jpeg')
#img_resized = cv2.resize(img, (700, 600))
img_resized = cv2.resize(img, (700, 700))

# THE DECTECTOR OBJECT HAS A BUILT IN FUNCTION TO DETECT FACES IN WHICH WE WILL PASS OUR IMAGE

detected_faces_op = detector.detect_faces(img_resized) #STORING THE DETECTED FACES IN A VARIABLE
print(detected_faces_op)


for i in detected_faces_op:

    # NOW WE WILL MAKE AN RECTANGLE ON FACE IN IMAGE USING OPENCV

    x,y,width,height = i['box'] # WE ARE ASSIGNING VALUES OF CO-ORDINATES,WIDTH AND HEIGHT FROM THE DICTIONARY HAVING KEY BOX FROM OUTPUT TO VARIABLES

    # EXTRACTING FACIAL LANDMARKS FROM KEYPOINT

    # LEFT AND RIGHT EYES

    x_leftEye,y_leftEye = i['keypoints']['left_eye']
    # USING CIRCLE TO POINT LEFT EYE 
    cv2.circle(img_resized,center=(x_leftEye,y_leftEye),color=(0,0,255),thickness=1,radius=2) # PARAMETER ARE IMAGE, CO-ORDINATES OF CENTER, COLOR, THICKNESS AND RADIUS IN PIXELS

    x_righttEye,y_rightEye = i['keypoints']['right_eye']
    cv2.circle(img_resized,center=(x_righttEye,y_rightEye),color=(0,0,255),thickness=1,radius=2)

    # NOSE 

    x_nose,y_nose = i['keypoints']['nose']
    cv2.circle(img_resized,center=(x_nose,y_nose),color=(0,0,255),thickness=1,radius=2)

    # LEFT MOUTH AND RIGHT MOUTH
    x_leftMouth,y_leftMouth = i['keypoints']['mouth_left']
    cv2.circle(img_resized,center=(x_leftMouth,y_leftMouth ),color=(0,0,255),thickness=1,radius=2)

    x_rightMouth,y_rightMouth = i['keypoints']['mouth_right']
    cv2.circle(img_resized,center=(x_rightMouth,y_rightMouth),color=(0,0,255),thickness=1,radius=2)

    #USING ABOVE VALUES WE WILL PRINT A RECTANGLE AROUND FACE IN AN IMAGE

    cv2.rectangle(img_resized,pt1=(x,y),pt2=(x+width,y+height),color=(255,0,0),thickness=3) # IN PARAMETERS WE ARE PASSING THE IMAGE, TWO POINTS TO DRAW RECTANGLE, COLOR OF RECTANGLE (BGR FORMAT) AND THICKNESS OF RECTANGLE

cv2.imshow('window',img_resized) #imshow A FUNCTION IN CV2 TO DISPLAY IMAGE HERE WE ARE USING WINDOW AND IMAGE PARAMETER
cv2.waitKey(0) # THIS ENABLES THE OUTPUT OF IMAGE TO STAY