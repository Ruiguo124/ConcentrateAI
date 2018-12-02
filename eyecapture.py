import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

eye =  {1:'Closed',0:'Open'}

eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
try:
    model = tf.keras.models.load_model('eye.h5')
except OSError as e: 
    print(e)
    exit()

# img = cv2.imread('open.jpg',cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img,(48,48))
# img = img / 255
# print(img)
        
#predictions = model.predict(img_pixels) 
# print(predictions[0])
# cv2.imshow('asd',img)
# cv2.waitKey(0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(147,20,255),2)
        

        detected_face = frame[int(y):int(y+h), int(x):int(x+w)] 
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) 
        detected_face = cv2.resize(detected_face, (48, 48)) 
        print(detected_face)
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        
        img_pixels /= 255 
        
        predictions = model.predict(img_pixels) 
        
        max_index = np.argmax(predictions[0])
        print(max_index)
        cv2.putText(frame, eye[int(max_index)], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)




        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
            
cap.release()
cv2.destroyAllWindows()