import cv2

# Load the Haar cascade classifiers for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define the threshold for closed eyes
eye_aspect_ratio_threshold = 0.3
eye_y_position_threshold = 0.4  # Adjust this value based on the position of the eyes in your setup

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Iterate over the detected eyes
    for (x, y, w, h) in eyes:
        # Draw rectangles around the detected eyes
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the region of interest (ROI) for the eye
        eye_roi = gray[y:y+h, x:x+w]
        
        # Calculate the aspect ratio of the eye
        eye_aspect_ratio = float(w) / h
        
        # Check if the aspect ratio is below the threshold and eye y-position is within the threshold range
        if eye_aspect_ratio < eye_aspect_ratio_threshold and y > eye_y_position_threshold * frame.shape[0]:
            print("Eyes are closed!")
        
    # Display the resulting frame
    cv2.imshow('Eye Tracker', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closed!")
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
