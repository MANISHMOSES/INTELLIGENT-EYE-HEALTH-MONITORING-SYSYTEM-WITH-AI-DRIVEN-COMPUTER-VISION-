import cv2
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import json
import tempfile # Import tempfile for using system temporary directory


# Placeholder imports for cloud storage libraries (uncomment and install as needed)
# from google.cloud import storage # For Google Cloud Storage
# import boto3 # For AWS S3
# from azure.storage.blob import BlobServiceClient # For Azure Blob Storage

class RecommendationMonitoring:
    """
    Manages generating health recommendations and alerts based on AI analysis results.
    """
    def __init__(self, user_info=None, alert_threshold=0.8):
        """
        Initializes the RecommendationMonitoring module.

        Args:
            user_info (dict, optional): Dictionary containing user information (e.g., {'age': 40}). Defaults to None.
            alert_threshold (float, optional): Confidence threshold for sending alerts (between 0 and 1). Defaults to 0.8.
        """
        self.user_info = user_info
        self.alert_threshold = alert_threshold

    def generate_recommendations(self, predicted_label, confidence):
        """
        Generates health suggestions based on AI analysis results.

        Args:
            predicted_label (str): The predicted eye health status (e.g., 'healthy', 'glaucoma', 'cataract', 'Model Error', 'Analysis Error', 'Empty ROI').
            confidence (float): The confidence level of the prediction (between 0 and 1).

        Returns:
            str: A recommendation message.
        """
        recommendation = "General Recommendation: Consult with an eye care professional for a comprehensive examination."

        if predicted_label == 'healthy':
            recommendation = "Recommendation: Maintain good eye health habits, including regular check-ups, balanced diet, and taking breaks during screen time."
        elif predicted_label == 'glaucoma':
            recommendation = "Recommendation: Based on the analysis, it is recommended to seek immediate consultation with an ophthalmologist for a glaucoma evaluation."
        elif predicted_label == 'cataract':
            recommendation = "Recommendation: It is advisable to consult with an eye specialist to discuss the potential for cataracts and treatment options."
        elif predicted_label in ["Model Error", "Analysis Error", "Empty ROI"]:
             recommendation = "Analysis incomplete/unsuccessful. Please ensure clear image/video quality and check system configurations."

        # Optionally, personalize recommendations based on user_info
        if self.user_info:
            if 'age' in self.user_info and self.user_info['age'] > 60:
                recommendation += " Age-related eye changes are common; regular check-ups are especially important."

        return recommendation

    def send_alert(self, predicted_label, confidence):
        """
        Determines if an alert should be sent based on AI analysis results and confidence.

        Args:
            predicted_label (str): The predicted eye health status.
            confidence (float): The confidence level of the prediction.

        Returns:
            str or None: An alert message if an alert is triggered, otherwise None.
        """
        alert_message = None
        # Only send alerts for specific health issues, not errors or empty ROIs
        if predicted_label != 'healthy' and predicted_label not in ["Model Error", "Analysis Error", "Empty ROI"]:
            if confidence >= self.alert_threshold:
                alert_message = f"ALERT: Potential severe eye health issue detected ({predicted_label}) with high confidence ({confidence:.2f}). Immediate medical attention is recommended."
            elif confidence > 0: # Alert for lower confidence but still non-healthy prediction
                 alert_message = f"NOTE: Potential eye health issue detected ({predicted_label}) with confidence ({confidence:.2f}). Further evaluation is recommended."

        return alert_message


class CloudDataManager:
    """
    Placeholder class for managing data storage in a cloud service.
    Replace placeholder methods with actual cloud provider SDK calls.
    """
    def __init__(self, cloud_provider=None, config=None):
        """
        Placeholder Initializes the Cloud Data Management module.

        Args:
            cloud_provider (str, optional): The name of the cloud provider (e.g., 'gcs', 's3', 'azure'). Defaults to None.
            config (dict, optional): Configuration details for the cloud provider (e.g., bucket name, credentials). Defaults to None.
        """
        self.cloud_provider = cloud_provider
        self.config = config
        self.client = None # Initialize client to None
        self._initialize_client()

    def _initialize_client(self):
        """
        Initializes the cloud storage client based on the configured provider.
        This is a placeholder method; actual implementation depends on the chosen cloud service.
        Uncomment and implement the logic for your chosen cloud provider.
        """
        if not self.cloud_provider or not self.config:
            print("Cloud provider or configuration missing. Cloud data management features will be unavailable.")
            self.client = None
            return

        print(f"Attempting to initialize {self.cloud_provider.upper()} cloud storage client (placeholder)...")

        try:
            if self.cloud_provider == 'gcs':
                # TODO: Implement GCS client initialization using google.cloud.storage
                # try:
                #     # Ensure google-cloud-storage is installed: pip install google-cloud-storage
                #     from google.cloud import storage
                #     self.client = storage.Client.from_service_account_json(self.config.get('credentials_path'))
                #     print("GCS client initialized (placeholder).")
                #     self.client = True # Simulate successful initialization
                # except ImportError:
                #     print("Google Cloud Storage library not installed. Please install it (`pip install google-cloud-storage`).")
                #     self.client = None
                # except Exception as e:
                #     print(f"Error initializing GCS client: {e}")
                #     self.client = None
                print("GCS client initialization placeholder executed.")
                self.client = True # Simulate successful initialization
                pass
            elif self.cloud_provider == 's3':
                # TODO: Implement AWS S3 client initialization using boto3
                # try:
                #     # Ensure boto3 is installed: pip install boto3
                #     import boto3
                #     self.client = boto3.client('s3',
                #                                aws_access_key_id=self.config.get('access_key_id'),
                #                                aws_secret_access_key=self.config.get('secret_access_key'),
                #                                region_name=self.config.get('region_name'))
                #     print("AWS S3 client initialized (placeholder).")
                #     self.client = True # Simulate successful initialization
                # except ImportError:
                #     print("boto3 library not installed. Please install it (`pip install boto3`).")
                #     self.client = None
                # except Exception as e:
                #     print(f"Error initializing S3 client: {e}")
                #     self.client = None
                print("AWS S3 client initialization placeholder executed.")
                self.client = True # Simulate successful initialization
                pass
            elif self.cloud_provider == 'azure':
                # TODO: Implement Azure Blob Storage client initialization using azure.storage.blob
                # try:
                #     # Ensure azure-storage-blob is installed: pip install azure-storage-blob
                #     from azure.storage.blob import BlobServiceClient
                #     self.client = BlobServiceClient.from_connection_string(self.config.get('connection_string'))
                #     print("Azure client initialized (placeholder).")
                #     self.client = True # Simulate successful initialization
                # except ImportError:
                #     print("Azure Blob Storage library not installed. Please install it (`pip install azure-storage-blob`).")
                #     self.client = None
                # except Exception as e:
                #     print(f"Error initializing Azure client: {e}")
                #     self.client = None
                print("Azure Blob Storage client initialization placeholder executed.")
                self.client = True # Simulate successful initialization
                pass
            else:
                print(f"Unsupported cloud provider specified: {self.cloud_provider}")
                self.client = None

            if self.client:
                print(f"✅ {self.cloud_provider.upper()} client initialized (placeholder).")
            else:
                 print(f"❌ Failed to initialize {self.cloud_provider.upper()} client (placeholder). Check configuration.")

        except Exception as e:
            print(f"Error during cloud client initialization (placeholder): {e}")
            self.client = None


    def upload_data(self, local_file_path, cloud_destination_path):
        """
        Placeholder Uploads data from a local file path to the specified cloud destination path.
        Replace with actual cloud provider SDK calls.

        Args:
            local_file_path (str): The path to the local file to upload.
            cloud_destination_path (str): The destination path (e.g., bucket/folder/file) in the cloud storage.

        Returns:
            bool: True if upload is successful (placeholder), False otherwise.
        """
        if self.client is None:
            print(f"Cloud client not initialized for {self.cloud_provider}. Cannot upload data from {local_file_path}.")
            return False

        if not os.path.exists(local_file_path):
            print(f"Error: Local file not found for upload: {local_file_path}")
            return False


        print(f"Attempting to upload data from {local_file_path} to {cloud_destination_path} (placeholder).")

        try:
            if self.cloud_provider == 'gcs':
                # TODO: Implement GCS upload using self.client and self.config.get('bucket_name')
                # bucket = self.client.get_bucket(self.config.get('bucket_name'))
                # blob = bucket.blob(cloud_destination_path)
                # blob.upload_from_filename(local_file_path)
                print("GCS upload placeholder executed.")
                time.sleep(0.1) # Simulate upload time
                pass
            elif self.cloud_provider == 's3':
                # TODO: Implement AWS S3 upload using self.client and self.config.get('bucket_name')
                # self.client.upload_file(local_file_path, self.config.get('bucket_name'), cloud_destination_path)
                print("S3 upload placeholder executed.")
                time.sleep(0.1) # Simulate upload time
                pass
            elif self.cloud_provider == 'azure':
                # TODO: Implement Azure Blob Storage upload using self.client and self.config.get('container_name')
                # blob_client = self.client.get_blob_client(container=self.config.get('container_name'), blob=cloud_destination_path)
                # with open(local_file_path, "rb") as data:
                #     blob_client.upload_blob(data)
                print("Azure upload placeholder executed.")
                time.sleep(0.1) # Simulate upload time
                pass
            # Assuming placeholder execution reaches here for success simulation
            print(f"Upload of {local_file_path} successful (placeholder).")
            return True # Placeholder success
        except Exception as e:
            print(f"Error during upload (placeholder): {e}")
            return False

    def download_data(self, cloud_source_path, local_destination_path):
        """
        Placeholder Downloads data from the specified cloud source path to a local destination path.
        Replace with actual cloud provider SDK calls.

        Args:
            cloud_source_path (str): The source path (e.g., bucket/folder/file) in the cloud storage.
            local_destination_path (str): The local file path to save the downloaded data.

        Returns:
            bool: True if download is successful (placeholder), False otherwise.
        """
        if self.client is None:
            print(f"Cloud client not initialized for {self.cloud_provider}. Cannot download data from {cloud_source_path}.")
            return False

        print(f"Attempting to download data from {cloud_source_path} to {local_destination_path} (placeholder).")
        try:
            local_dir = os.path.dirname(local_destination_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir)
                print(f"Created local directory: {local_dir}")
            # TODO: Implement actual download based on self.cloud_provider
            time.sleep(0.1) # Simulate download time
            print(f"Download of {cloud_source_path} successful (placeholder).")
            return True
        except Exception as e:
            print(f"Error during download (placeholder): {e}")
            return False

    def list_files(self, cloud_path):
        """
        Placeholder Lists files in a specified path within the cloud storage.
        Replace with actual cloud provider SDK calls.

        Args:
            cloud_path (str): The path (e.g., bucket/folder) in the cloud storage to list files from.

        Returns:
            list: A list of file names or objects (placeholder), or None if an error occurs.
        """
        if self.client is None:
            print(f"Cloud client not initialized for {self.cloud_provider}. Cannot list files in {cloud_path}.")
            return None
        print(f"Attempting to list files in {cloud_path} (placeholder).")
        try:
            # TODO: Implement actual list files based on self.cloud_provider
            time.sleep(0.1) # Simulate list time
            print(f"List files in {cloud_path} successful (placeholder).")
            return [f"placeholder_file_{i}.txt" for i in range(3)] # Placeholder result
        except Exception as e:
            print(f"Error during listing files (placeholder): {e}")
            return None


    def delete_data(self, cloud_path):
        """
        Placeholder Deletes data from the specified path in the cloud storage.
        Replace with actual cloud provider SDK calls.

        Args:
            cloud_path (str): The path (e.g., bucket/folder/file) in the cloud storage to delete.

        Returns:
            bool: True if deletion is successful (placeholder), False otherwise.
        """
        if self.client is None:
            print(f"Cloud client not initialized for {self.cloud_provider}. Cannot delete data at {cloud_path}.")
            return False
        print(f"Attempting to delete data at {cloud_path} (placeholder).")
        try:
            # TODO: Implement actual delete based on self.cloud_provider
            time.sleep(0.1) # Simulate delete time
            print(f"Deletion of {cloud_path} successful (placeholder).")
            return True
        except Exception as e:
            print(f"Error during deletion (placeholder): {e}")
            return False


class EyeHealthMonitor:
    """
    Integrates visual data collection, AI analysis, recommendation, and cloud data management
    for eye health monitoring. Supports both live camera monitoring and file processing.
    """
    def __init__(self, camera_source=0, model_path='eye_health_cnn_model.h5', user_info=None, alert_threshold=0.8, cloud_config=None):
        """
        Initializes the Eye Health Monitor.

        Args:
            camera_source (int): The index of the camera to use (0, 1, etc.). Defaults to 0.
            model_path (str): The path to the trained CNN model file (.h5 or .keras).
            user_info (dict, optional): Dictionary containing user information for recommendations. Defaults to None.
            alert_threshold (float, optional): Confidence threshold for sending alerts. Defaults to 0.8.
            cloud_config (dict, optional): Dictionary containing cloud provider configuration.
                                           Expected keys: 'provider' (str, e.g., 'gcs', 's3', 'azure'),
                                                          'config' (dict, provider-specific configuration).
                                           Set to None to disable cloud storage. Defaults to None.
        """
        self.camera = None
        self.is_capturing = False
        self.camera_source = camera_source
        self.cloud_config = cloud_config

        # Load the pre-trained Haar Cascade classifier for eye detection
        # The haarcascade_eye.xml file is usually included with OpenCV installations
        # Ensure it's accessible in the OpenCV data directory or provide the full path
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        if self.eye_cascade.empty():
            print("Error: Eye cascade classifier not loaded.")
            print("Please ensure 'haarcascade_eye.xml' is in your OpenCV data directory.")
            # Consider adding a mechanism to exit or handle this error gracefully

        # Load the pre-trained CNN model and compile it manually
        self.cnn_model = None
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
        else:
            try:
                # Load the model
                self.cnn_model = load_model(model_path)
                print(f"CNN model loaded successfully from {model_path}.")

                # Manually compile the model
                # Use the optimizer and loss function that were used during training
                # Ensure these match the ones used during training for correct behavior
                self.cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                                       loss=tf.keras.losses.CategoricalCrossentropy(),
                                       metrics=['accuracy'])
                print("CNN model compiled successfully.")

            except Exception as e:
                print(f"Error loading or compiling CNN model from {model_path}: {e}")
                print("Please ensure the model file is valid and the compile parameters match training.")


        # Define image dimensions for the model (should match training)
        # Assuming the model was trained on 128x128 grayscale images
        self.img_height = 128
        self.img_width = 128
        # Define class labels based on your training data
        # This needs to match the classes the model was trained on
        self.class_labels = ['healthy', 'glaucoma', 'cataract'] # Example: Replace with actual class labels trained on

        # Initialize RecommendationMonitoring module
        self.recommendation_monitor = RecommendationMonitoring(user_info=user_info, alert_threshold=alert_threshold)

        # Initialize CloudDataManager module if cloud_config is provided and valid
        self.cloud_data_manager = None
        if self.cloud_config and isinstance(self.cloud_config, dict) and 'provider' in self.cloud_config and 'config' in self.cloud_config:
            self.cloud_data_manager = CloudDataManager(cloud_provider=self.cloud_config['provider'],
                                                       config=self.cloud_config['config'])
            if self.cloud_data_manager and self.cloud_data_manager.client is None:
                 print("Cloud data manager initialization failed. Cloud storage will be disabled.")
                 self.cloud_data_manager = None # Disable if initialization failed

        else:
            print("Cloud storage is disabled or configuration is incomplete/invalid.")


    def initialize_camera(self):
        """
        Initializes the camera or video source by trying multiple indices.

        Returns:
            bool: True if a camera is successfully initialized, False otherwise.
        """
        print("🔍 Searching for available cameras...")
        for i in range(3):  # Try camera sources 0, 1, 2
            # Use cv2.CAP_DSHOW backend on Windows for potentially faster initialization
            # Remove or change if on a different OS or specific backend is needed
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # For Windows
            except: # Fallback for non-Windows or if CAP_DSHOW is not needed/available
                cap = cv2.VideoCapture(i)

            if cap.isOpened():
                print(f"✅ Camera initialized successfully at source {i}.")
                self.camera = cap
                self.camera_source = i
                self.is_capturing = True
                return True
            else:
                print(f"❌ Could not open camera with source {i}.")
                if cap: # Release if it was opened but not successfully
                    cap.release() # Ensure resource is released

        print("🚫 Error: Could not open any camera (sources 0, 1, 2).")
        self.camera = None
        return False


    def capture_frame(self):
        """
        Captures a single frame from the camera.

        Returns:
            numpy.ndarray or None: The captured frame as a NumPy array, or None if capture failed.
        """
        if self.camera and self.is_capturing:
            ret, frame = self.camera.read()
            if not ret:
                print("Warning: Could not read frame from camera.")
                return None
            return frame
        else:
            # print("Camera not initialized or capturing is off.") # Suppress repeated message
            return None

    def analyze_eye(self, eye_roi):
        """
        Preprocesses an extracted eye region and performs CNN prediction.

        Args:
            eye_roi (numpy.ndarray): The image region containing the detected eye.

        Returns:
            tuple: A tuple containing the predicted label (str) and confidence (float).
                   Returns ("Model Error", 0.0), ("Empty ROI", 0.0), or ("Analysis Error", 0.0) on failure.
        """
        if self.cnn_model is None:
            return "Model Error", 0.0 # Return error if model not loaded

        # Validate input eye region
        if eye_roi is None or eye_roi.shape[0] == 0 or eye_roi.shape[1] == 0:
             return "Empty ROI", 0.0
        # Ensure the ROI has 3 channels (color) before converting to grayscale
        if len(eye_roi.shape) == 2: # If grayscale already
             processed_roi = eye_roi
        else: # Assume BGR color
            processed_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        try:
            # Resize to model input dimensions
            resized_eye = cv2.resize(processed_roi, (self.img_width, self.img_height))
            # Normalize pixel values to [0, 1]
            normalized_eye = resized_eye.astype('float32') / 255.0
            # Reshape for the model (add batch dimension and channel dimension for grayscale [1, H, W, 1])
            preprocessed_eye = np.expand_dims(normalized_eye, axis=0)
            preprocessed_eye = np.expand_dims(preprocessed_eye, axis=-1)

            # Perform CNN prediction
            # verbose=0 suppresses the progress bar from predict
            predictions = self.cnn_model.predict(preprocessed_eye, verbose=0)
            predicted_class_index = np.argmax(predictions)
            predicted_label = self.class_labels[predicted_class_index]
            confidence = predictions[0][predicted_class_index]
            return predicted_label, confidence
        except Exception as e:
            print(f"Error during eye analysis: {e}")
            return "Analysis Error", 0.0


    def start_monitoring(self):
        """
        Starts continuous camera monitoring, performs eye detection and analysis,
        and displays results and recommendations on the live feed.
        Press 'q' to stop monitoring.
        """
        if not self.initialize_camera():
             print("Failed to start monitoring due to camera initialization error.")
             return

        if self.eye_cascade.empty():
             print("Monitoring stopped: Eye cascade classifier not loaded.")
             self.release_camera()
             return
        if self.cnn_model is None:
             print("Monitoring stopped: CNN model not loaded.")
             self.release_camera()
             return


        print("Starting continuous monitoring. Press 'q' to stop.")
        while self.is_capturing:
            frame = self.capture_frame()
            if frame is not None:
                # Convert the frame to grayscale for eye detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect eyes in the grayscale frame
                eyes = self.eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                overall_recommendation = "" # Initialize recommendation message for the frame
                analysis_results_frame = [] # Store results for potential upload (if needed for live)

                # Process each detected eye
                if len(eyes) > 0:
                    for (x, y, w, h) in eyes:
                        # Extract the eye region - ensure correct slicing
                        # Add a small padding around the detected eye for better analysis
                        padding = 10 # Adjust padding as needed
                        y1, y2, x1, x2 = max(0, y - padding), min(frame.shape[0], y + h + padding), max(0, x - padding), min(frame.shape[1], x + w + padding)
                        eye_roi = frame[y1:y2, x1:x2]


                        # Analyze the eye region using the CNN model
                        predicted_label, confidence = self.analyze_eye(eye_roi)

                        # Store analysis result for this eye in this frame
                        analysis_results_frame.append({
                            'timestamp': time.time(), # Timestamp for live feed
                            'bbox': (int(x), int(y), int(w), int(h)),
                            'predicted_label': predicted_label,
                            'confidence': float(confidence) if not isinstance(confidence, str) else confidence # Convert numpy float to standard float
                        })


                        # Define color based on predicted health status
                        color = (0, 255, 0) if predicted_label == 'healthy' else (0, 0, 255) # Green for healthy, Red otherwise

                        # Draw bounding box and label on the original frame
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        label = f"{predicted_label}: {confidence:.2f}" if not isinstance(confidence, str) else predicted_label
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        # Generate and potentially print recommendations/alerts for this eye
                        eye_recommendation = self.recommendation_monitor.generate_recommendations(predicted_label, confidence)
                        alert_message = self.recommendation_monitor.send_alert(predicted_label, confidence)

                        if alert_message:
                            print(alert_message) # Print alert to console

                        # Prioritize non-healthy recommendations for display
                        if predicted_label != 'healthy' and predicted_label not in ["Model Error", "Analysis Error", "Empty ROI"]:
                            overall_recommendation = eye_recommendation # Display the first non-healthy recommendation found
                        elif overall_recommendation == "" and predicted_label == 'healthy':
                             overall_recommendation = eye_recommendation # Display healthy recommendation if no non-healthy found yet


                # Display overall recommendation on the frame
                if overall_recommendation:
                     cv2.putText(frame, overall_recommendation, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # --- Cloud Data Management Integration (Live Monitoring - Optional) ---
                # In a real-time monitoring scenario, you might want to upload
                # analysis results periodically or when specific events occur (e.g., a non-healthy detection)
                # This is a placeholder. Uncomment and adapt as needed for your application's requirements.
                # if self.cloud_data_manager and analysis_results_frame:
                #     try:
                #         timestamp = int(time.time())
                #         results_filename = f"live_analysis_{timestamp}.json"
                #         results_filepath = os.path.join(tempfile.gettempdir(), results_filename) # Use system temp directory
                #         with open(results_filepath, 'w') as f:
                #             json.dump(analysis_results_frame, f, indent=4)
                #         cloud_dest_path = f"live_monitoring/{results_filename}"
                #         self.cloud_data_manager.upload_data(results_filepath, cloud_dest_path)
                #         os.remove(results_filepath) # Clean up temporary file
                #     except Exception as e:
                #         print(f"Error uploading live monitoring results: {e}")
                # ---------------------------------------------------------------------


                cv2.imshow("Eye Health Monitoring", frame)

                # Stop capture if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_monitoring()
            time.sleep(0.01) # Small delay to prevent high CPU usage and reduce CPU load


    def stop_monitoring(self):
        """ Stops the monitoring process. """
        self.is_capturing = False
        print("Monitoring stopped.")


    def release_camera(self):
        """ Releases the camera resources. """
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows() # Close all OpenCV windows
            print("Camera released.")


    def process_file(self, file_path):
        """
        Processes a pre-recorded video or image file for eye health analysis and recommendations.
        Also uploads the original file and analysis results to the cloud if configured.

        Args:
            file_path (str): The path to the video or image file.
        """
        if self.eye_cascade.empty():
             print("File processing stopped: Eye cascade classifier not loaded.")
             return
        if self.cnn_model is None:
             print("File processing stopped: CNN model not loaded.")
             return

        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return

        # Determine if it's a video or image file
        is_video = False
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            # Check if it's truly a video and not just a readable image file
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 1:
                is_video = True
            else:
                cap.release() # Release if it's not a video

        frame = None # Initialize frame for image processing case
        if not is_video:
            # Try reading as an image if not a video
            frame = cv2.imread(file_path)
            if frame is None:
                print(f"Error: Could not read file as video or image: {file_path}")
                return

        print(f"Processing file: {file_path}")

        analysis_results_list = [] # Store analysis results for the entire file

        if is_video:
            print("Processing video file. Press 'q' to stop.")
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                frame_count += 1
                # Convert the frame to grayscale for eye detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect eyes in the grayscale frame
                eyes = self.eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                overall_recommendation = "" # Initialize recommendation message for the frame
                frame_analysis_results = [] # Store results for eyes in this frame

                # Process each detected eye
                if len(eyes) > 0:
                    for (x, y, w, h) in eyes:
                        # Extract the eye region - ensure correct slicing
                        # Add a small padding around the detected eye for better analysis
                        padding = 10 # Adjust padding as needed
                        y1, y2, x1, x2 = max(0, y - padding), min(frame.shape[0], y + h + padding), max(0, x - padding), min(frame.shape[1], x + w + padding)
                        eye_roi = frame[y1:y2, x1:x2]


                        # Analyze the eye region using the CNN model
                        predicted_label, confidence = self.analyze_eye(eye_roi)

                        # Store analysis result for this eye and frame
                        frame_analysis_results.append({
                            'frame': frame_count,
                            'bbox': (int(x), int(y), int(w), int(h)),
                            'predicted_label': predicted_label,
                            'confidence': float(confidence) if not isinstance(confidence, str) else confidence
                        })


                        # Define color based on predicted health status
                        color = (0, 255, 0) if predicted_label == 'healthy' else (0, 0, 255) # Green for healthy, Red otherwise

                        # Draw bounding box and label on the frame
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        label = f"{predicted_label}: {confidence:.2f}" if not isinstance(confidence, str) else predicted_label
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        # Generate and potentially print recommendations/alerts for this eye
                        eye_recommendation = self.recommendation_monitor.generate_recommendations(predicted_label, confidence)
                        alert_message = self.recommendation_monitor.send_alert(predicted_label, confidence)

                        if alert_message:
                            print(alert_message) # Print alert to console

                        # Prioritize non-healthy recommendations for display
                        if predicted_label != 'healthy' and predicted_label not in ["Model Error", "Analysis Error", "Empty ROI"]:
                            overall_recommendation = eye_recommendation # Display the first non-healthy recommendation found
                        elif overall_recommendation == "" and predicted_label == 'healthy':
                             overall_recommendation = eye_recommendation # Display healthy recommendation if no non-healthy found yet

                # Append results for this frame to the overall list
                if frame_analysis_results:
                    analysis_results_list.extend(frame_analysis_results)


                # Display overall recommendation on the frame
                if overall_recommendation:
                     cv2.putText(frame, overall_recommendation, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


                cv2.imshow("File Analysis", frame)

                # Stop processing if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release() # Release video capture object

        else: # Process as a single image
            print("Processing image file.")
            # Frame is already loaded

            # Convert the frame to grayscale for eye detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect eyes in the grayscale frame
            eyes = self.eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            overall_recommendation = "" # Initialize recommendation message for the frame
            image_analysis_results = [] # Store results for eyes in this image

            # Process each detected eye
            if len(eyes) > 0:
                for (x, y, w, h) in eyes:
                    # Extract the eye region
                    # Add a small padding around the detected eye for better analysis
                    padding = 10 # Adjust padding as needed
                    y1, y2, x1, x2 = max(0, y - padding), min(frame.shape[0], y + h + padding), max(0, x - padding), min(frame.shape[1], x + w + padding)
                    eye_roi = frame[y1:y2, x1:x2]


                    # Analyze the eye region using the CNN model
                    predicted_label, confidence = self.analyze_eye(eye_roi)

                    # Store analysis result for this eye and image
                    image_analysis_results.append({
                        'bbox': (int(x), int(y), int(w), int(h)),
                        'predicted_label': predicted_label,
                        'confidence': float(confidence) if not isinstance(confidence, str) else confidence
                    })

                    # Define color based on predicted health status
                    color = (0, 255, 0) if predicted_label == 'healthy' else (0, 0, 255) # Green for healthy, Red otherwise

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    label = f"{predicted_label}: {confidence:.2f}" if not isinstance(confidence, str) else predicted_label
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Generate and potentially print recommendations/alerts for this eye
                    eye_recommendation = self.recommendation_monitor.generate_recommendations(predicted_label, confidence)
                    alert_message = self.recommendation_monitor.send_alert(predicted_label, confidence)

                    if alert_message:
                        print(alert_message) # Print alert to console

                    # Prioritize non-healthy recommendations for display
                    if predicted_label != 'healthy' and predicted_label not in ["Model Error", "Analysis Error", "Empty ROI"]:
                        overall_recommendation = eye_recommendation # Display the first non-healthy recommendation found
                    elif overall_recommendation == "" and predicted_label == 'healthy':
                         overall_recommendation = eye_recommendation # Display healthy recommendation if no non-healthy found yet


            # Display overall recommendation on the frame
            if overall_recommendation:
                 cv2.putText(frame, overall_recommendation, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


            cv2.imshow("File Analysis", frame)

            # Wait for a key press ('q') or any other key to close the image window
            print("Press 'q' or any other key to close the image window.")
            if cv2.waitKey(0) & 0xFF == ord('q'):
                pass # If 'q' is pressed, just continue to potentially close windows
            cv2.destroyAllWindows() # Close image window after key press


        print("File processing complete.")

        # --- Cloud Data Management Integration (File Processing) ---
        if self.cloud_data_manager and analysis_results_list:
            print("Attempting to upload analysis results and original file to cloud.")
            timestamp = int(time.time())
            original_file_name = os.path.basename(file_path)
            # Create a unique filename for results to avoid conflicts
            results_filename = f"{os.path.splitext(original_file_name)[0]}_analysis_results_{timestamp}.json"
            # Use a temporary directory for saving the results file before upload
            temp_dir = tempfile.gettempdir() # Get the system's temporary directory
            results_filepath_local = os.path.join(temp_dir, results_filename)

            try:
                # Save analysis results to a local JSON file
                with open(results_filepath_local, 'w') as f:
                    json.dump(analysis_results_list, f, indent=4)
                print(f"Analysis results saved locally to {results_filepath_local}")

                # Define cloud destination paths
                cloud_results_dest_path = f"analysis_results/{results_filename}"
                cloud_original_file_dest_path = f"original_files/{original_file_name}"

                # Upload the analysis results file
                upload_results_success = self.cloud_data_manager.upload_data(results_filepath_local, cloud_results_dest_path)

                # Upload the original file
                upload_original_success = self.cloud_data_manager.upload_data(file_path, cloud_original_file_dest_path)

                # Clean up local temporary results file if upload was attempted
                if upload_results_success: # Only remove if results file was successfully uploaded (or upload attempted placeholder)
                     try:
                         os.remove(results_filepath_local)
                         print(f"Cleaned up temporary results file: {results_filepath_local}")
                     except OSError as e:
                         print(f"Error removing temporary file {results_filepath_local}: {e}")


            except Exception as e:
                print(f"Error during cloud upload process after file processing: {e}")
        elif self.cloud_data_manager and not analysis_results_list:
             print("No eye analysis results to upload for this file.")
        else:
             print("Cloud data manager not initialized or configured. Skipping cloud upload.")
        # -----------------------------------------------------------


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # Replace 'path/to/your/sample_image_or_video.file' with the actual path to your file
    # Example: 'sample_image.jpg' or 'sample_video.mp4'
    file_to_process = 'path/to/your/sample_image_or_video.file'
    # Replace 'path/to/your/eye_health_cnn_model.h5' with the actual path to your trained model
    model_file_path = 'eye_health_cnn_model.h5' # Assuming the model is in the same directory

    user_profile = {'age': 40} # Example user info for personalized recommendations
    alert_confidence_threshold = 0.75 # Confidence threshold for triggering 'ALERT' level notifications (between 0 and 1)

    # --- Cloud Storage Configuration (Optional) ---
    # Uncomment and configure the cloud_storage_config dictionary if you want to enable cloud storage.
    # Make sure to replace the placeholder values with your actual cloud service details.
    # Ensure you have the necessary cloud SDKs installed (e.g., `pip install google-cloud-storage boto3 azure-storage-blob`).
    # IMPORTANT: Handle credentials securely in a production environment (e.g., environment variables, IAM roles).
    #
    # Example configuration for Google Cloud Storage (GCS):
    # cloud_storage_config = {
    #     'provider': 'gcs',
    #     'config': {
    #         'bucket_name': 'your-gcs-bucket-name',
    #         'credentials_path': 'path/to/your/gcs/credentials.json' # Path to your service account key file
    #     }
    # }
    #
    # Example configuration for AWS S3:
    # cloud_storage_config = {
    #     'provider': 's3',
    #     'config': {
    #         'bucket_name': 'your-s3-bucket-name',
    #         'access_key_id': 'YOUR_ACCESS_KEY_ID',
    #         'secret_access_key': 'YOUR_SECRET_ACCESS_KEY',
    #         'region_name': 'your-region' # e.g., 'us-east-1'
    #     }
    # }
    #
    # Example configuration for Azure Blob Storage:
    # cloud_storage_config = {
    #     'provider': 'azure',
    #     'config': {
    #         'container_name': 'your-azure-container-name',
    #         'connection_string': 'YOUR_AZURE_STORAGE_CONNECTION_STRING'
    #     }
    # }

    cloud_storage_config = None # Set to None to disable cloud storage functionality


    # Initialize the monitor for file processing
    # Pass cloud_config if you want to enable cloud storage
    monitor = EyeHealthMonitor(model_path=model_file_path,
                               user_info=user_profile,
                               alert_threshold=alert_confidence_threshold,
                               cloud_config=cloud_storage_config)

    # Process the file
    # This will load the file, detect eyes, analyze, display results,
    # and potentially upload data to the cloud.
    monitor.process_file(file_to_process)

    # --- To run live camera monitoring instead, uncomment the following lines and comment out the file processing lines above ---
    # Note: Live camera monitoring requires a connected and accessible camera.
    # monitor = EyeHealthMonitor(camera_source=0, model_path=model_file_path, user_info=user_profile, alert_threshold=alert_confidence_threshold)
    # monitor.start_monitoring()
    # monitor.release_camera() # Ensure camera is released when monitoring stops
