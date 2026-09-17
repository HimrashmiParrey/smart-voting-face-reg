import streamlit as st
import cv2
import numpy as np
import pickle
import os

# Get the absolute directory path of where app.py is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure data directory exists inside our project folder
data_dir = os.path.join(BASE_DIR, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

st.title("Smart Voting System - Face Registration")
st.write("Register your face to participate in the voting process.")

# Input for Student Code
name = st.text_input("Enter your Student Code:", placeholder="e.g., STU12345")

# Camera input widget
img_file_buffer = st.camera_input("Take a photo of your face")

# Create a robust, absolute path to the XML file
cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

if img_file_buffer is not None and name:
    # Convert the image buffer to an OpenCV image
    bytes_data = img_file_buffer.getvalue()
    cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Check if cascade file exists before loading
    if os.path.exists(cascade_path):
        # Load face cascade safely
        facedetect = cv2.CascadeClassifier(cascade_path)
        
        # Verify the classifier loaded correctly
        if facedetect.empty():
            st.error("Error: Could not load the face detection XML structure. Please redeploy.")
        else:
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                st.error("No face detected! Please adjust your lighting or angle and try again.")
            else:
                # Take the first detected face
                # detectMultiScale returns a list of faces or a single face array depending on match count
                # Let's handle it safely by pulling the first index
                (x, y, w, h) = faces[0] if len(faces.shape) > 1 else faces
                crop_img = cv_image[y:y+h, x:x+w]
                resized_img = cv2.resize(crop_img, (50, 50))
                
                # Prepare data to save (flatten to match original format)
                faces_data = np.asarray([resized_img])
                faces_data = faces_data.reshape((1, -1))
                
                # --- Save Name ---
                names_path = os.path.join(data_dir, 'names.pk1')
                if not os.path.exists(names_path):
                    names = [name]
                else:
                    with open(names_path, 'rb') as f:
                        names = pickle.load(f)
                    names.append(name)
                    
                with open(names_path, 'wb') as f:
                    pickle.dump(names, f)
                    
                # --- Save Face Data ---
                faces_path = os.path.join(data_dir, 'faces_data.pk1')
                if not os.path.exists(faces_path):
                    with open(faces_path, 'wb') as f:
                        pickle.dump(faces_data, f)
                else:
                    with open(faces_path, 'rb') as f:
                        existing_faces = pickle.load(f)
                    updated_faces = np.append(existing_faces, faces_data, axis=0)
                    with open(faces_path, 'wb') as f:
                        pickle.dump(updated_faces, f)
                
                st.success(f"Success! Face registered for student code: {name}")
    else:
        st.error(f"Cascade file missing! Make sure 'haarcascade_frontalface_default.xml' is uploaded to your GitHub repository root folder.")
elif img_file_buffer is not None and not name:
    st.warning("Please enter your Student Code before taking the photo.")
