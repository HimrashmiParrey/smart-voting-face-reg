import streamlit as st
import cv2
import numpy as np
import pickle
import os
import urllib.request

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

# Fallback: Download the XML directly from OpenCV's official source to ensure validity
@st.cache_resource
def load_cascade():
    url = "https://githubusercontent.com"
    xml_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
    
    # Download if it doesn't exist or is empty
    if not os.path.exists(xml_path) or os.path.getsize(xml_path) == 0:
        try:
            urllib.request.urlretrieve(url, xml_path)
        except Exception as e:
            st.error(f"Failed to fetch face detection files from the internet: {e}")
            return None
            
    detector = cv2.CascadeClassifier(xml_path)
    if detector.empty():
        return None
    return detector

facedetect = load_cascade()

if img_file_buffer is not None and name:
    if facedetect is None:
        st.error("Face detector configuration error. Please check your internet connection or redeploy the app.")
    else:
        # Convert the image buffer to an OpenCV image
        bytes_data = img_file_buffer.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            st.error("No face detected! Please adjust your lighting or angle and try again.")
        else:
            # Handle shape mismatch safely if multiple faces or structural single faces are returned
            first_face = faces[0] if isinstance(faces, np.ndarray) and len(faces.shape) > 1 else faces
            
            # Extract coordinates safely
            try:
                if len(first_face) == 4:
                    x, y, w, h = first_face
                else:
                    x, y, w, h = faces[0][0], faces[0][1], faces[0][2], faces[0][3]
                    
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
            except Exception as e:
                st.error(f"Error processing face coordinates. Please try standing closer to the camera.")
                
elif img_file_buffer is not None and not name:
    st.warning("Please enter your Student Code before taking the photo.")

