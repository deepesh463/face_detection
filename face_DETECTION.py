import cv2

# cascade reqiured
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade =cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   # detecting the face 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
    # detecting the eyes    
        eyes = eyes_cascade.detectMultiScale(roi_gray,1.1,7)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
     
    # detecting the smile
        smile = smile_cascade.detectMultiScale(roi_gray ,1.8,20)        
        for (esx,esy,esw,esh) in smile :
             cv2.rectangle(roi_color,(esx,esy),(esx+esw,esy+esh),(0,0,255),2)

    cv2.imshow('Video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
