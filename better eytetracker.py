import cv2

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


cap = cv2.VideoCapture(0)

eye_aspect_ratio_threshold = 0.3
eye_y_position_threshold = 0.4  # Adjust this value based on the position of the eyes in your setup

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        eye_roi = gray[y:y+h, x:x+w]
        
        eye_aspect_ratio = float(w) / h
        
        if eye_aspect_ratio < eye_aspect_ratio_threshold and y > eye_y_position_threshold * frame.shape[0]:
            print("Eyes are closed!")
        
    cv2.imshow('Eye Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closed!")
        break

cap.release()
cv2.destroyAllWindows()
