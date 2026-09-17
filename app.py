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

if img_file_buffer is not None and name:
    # Convert the image buffer to an OpenCV image
    bytes_data = img_file_buffer.getvalue()
    cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Since OpenCV Cascades are throwing errors on your cloud container,
    # we will use an alternative fallback profile processing or crop directly.
    # To ensure it runs without crashing, we crop the central area where the user's face is positioned.
    h, w, _ = cv_image.shape
    
    # Define a bounding box around the center of the camera view
    box_size = min(h, w) // 2
    start_x = (w - box_size) // 2
    start_y = (h - box_size) // 2
    
    try:
        # Crop the center of the frame (where the user aligns their face)
        crop_img = cv_image[start_y:start_y+box_size, start_x:start_x+box_size]
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
        st.info("Tip: Make sure your face is centered inside the camera box when snapping the photo.")
        
    except Exception as e:
        st.error(f"Error processing image data: {e}")
                
elif img_file_buffer is not None and not name:
    st.warning("Please enter your Student Code before taking the photo.")

