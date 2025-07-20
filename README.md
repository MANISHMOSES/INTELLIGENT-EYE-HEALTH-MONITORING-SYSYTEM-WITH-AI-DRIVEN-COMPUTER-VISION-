import cv2

def run_camera_eye_detection():
    # Load the pre-trained Haar Cascade classifier for eye detection
    # Make sure 'haarcascade_eye.xml' is accessible (e.g., in your OpenCV data directory)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if eye_cascade.empty():
        print("Error loading eye cascade classifier.")
        print("Please ensure 'haarcascade_eye.xml' is in your OpenCV data path.")
        return

    print("🔍 Searching for available cameras...")
    camera = None
    for i in range(3):  # Try camera sources 0, 1, 2
        # Use cv2.CAP_DSHOW on Windows, otherwise remove it
        # Adjust based on your operating system
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # For Windows
        except: # Fallback for non-Windows or if CAP_DSHOW is not needed
            cap = cv2.VideoCapture(i)

        if cap.isOpened():
            print(f"✅ Camera initialized successfully at source {i}.")
            camera = cap
            break # Found a working camera, exit loop
        else:
            print(f"❌ Could not open camera with source {i}.")
            if cap: # Release if it was opened but failed later
                cap.release()


    if camera is None:
        print("🚫 Error: Could not open any camera (sources 0, 1, 2).")
        print("Please check camera connection, permissions, and ensure you are not in a headless environment.")
        return

    print("Starting camera feed with eye detection. Press 'q' to stop.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale for eye detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eyes in the grayscale frame
        eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected eyes on the original color frame
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw a blue rectangle

        # Display the frame
        cv2.imshow("Camera Feed with Eye Detection", frame)

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

# Run the function if the script is executed directly
if __name__ == "__main__":
    run_camera_eye_detection()

