#importing the imortant libraires
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
#open webcam
cap=cv2.VideoCapture(0)

#load model
model=load_model("gender.model")
#open CacadeClassifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#loop for faces
while(cap.isOpened()):
    # read from webcam
    ret,frame=cap.read()
    
    #converting color to gray scale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #applying face detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)    
    
    #looping through detected faces
    for (x,y,w,h) in faces:
    	#draw rectangle around faces
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        #Preproces the detected faces
        face_crop = np.copy(frame[y:y+h,x:x+h])
        face_crop=cv2.resize(face_crop,(150,150))
        face_crop = image.img_to_array(face_crop)
        face_crop=face_crop.reshape(1,150,150,3)
        

        #predicting the faces     
        conf = model.predict(face_crop)
        
        #mapping the result
        if conf ==0:
            prediction='Man'
        else:
            prediction='Women'
            

        
        label = "{}".format(prediction)

        #putting the text on detected faces  with label 
        cv2.putText(frame,label,(x+10,y+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2, cv2.LINE_AA)
        

    #display the output
    cv2.imshow("gender",frame)
    # press "q" to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
#release resource        
cap.release()
cv2.destroyAllWindows()