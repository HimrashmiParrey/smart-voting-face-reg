import streamlit as st
import cv2
import numpy as np
import pickle
import os

# Ensure data directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

st.title("Smart Voting System - Face Registration")
st.write("Register your face to participate in the voting process.")

# Input for Student Code
name = st.text_input("Enter your Student Code:", placeholder="e.g., STU12345")

# Camera input widget
img_file_buffer = st.camera_input("Take a photo of your face")

# XML path - using a relative path is better for cloud servers
cascade_path = "haarcascade_frontalface_default.xml"

if img_file_buffer is not None and name:
    # Convert the image buffer to an OpenCV image
    bytes_data = img_file_buffer.getvalue()
    cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade
    if os.path.exists(cascade_path):
        facedetect = cv2.CascadeClassifier(cascade_path)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            st.error("No face detected! Please adjust your lighting or angle and try again.")
        else:
            # Take the first detected face
            (x, y, w, h) = faces[0]
            crop_img = cv_image[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50))
            
            # Prepare data to save (flatten to match your original format)
            faces_data = np.asarray([resized_img])
            faces_data = faces_data.reshape((1, -1))
            
            # --- Save Name ---
            names_path = 'data/names.pk1'
            if not os.path.exists(names_path):
                names = [name]
            else:
                with open(names_path, 'rb') as f:
                    names = pickle.load(f)
                names.append(name)
                
            with open(names_path, 'wb') as f:
                pickle.dump(names, f)
                
            # --- Save Face Data ---
            faces_path = 'data/faces_data.pk1'
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
        st.error(f"Cascade file missing! Make sure '{cascade_path}' is in the same directory.")
elif img_file_buffer is not None and not name:
    st.warning("Please enter your Student Code before taking the photo.")
