import streamlit as st
import os
import new
import time
import check

# Constants
UPLOAD_DIRECTORY = "temp"


def save_uploaded_file(uploaded_file):
    try:
        path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return path
    except Exception as e:
        st.write(e)
        return None


st.title("Video Action Classifier")

uploaded_file = st.file_uploader(
    "Choose a video file", type=["mp4", "mkv", "avi", "mov"]
)
if uploaded_file is not None:
    st.write("File successfully uploaded.")

    if st.button("Predict Action"):
        video_path = save_uploaded_file(uploaded_file)

        if video_path:
            start_time = time.time()
            class_belong, val = new.predict_single_action(video_path, 11)
            end_time = time.time()
            elapsed_time = end_time - start_time

            st.write(f"File Name: {uploaded_file.name}")
            st.write(f"Action Predicted: {class_belong}")
            st.write(f"Accuracy Score: {val}")
            st.write(f"Time delay for the prediction: {elapsed_time:.2f} seconds")

            check.output(uploaded_file.name)

            # You may need additional logic to display the 'middle_frame.jpg' as well
            image_path = os.path.join("static", "middle_frame.jpg")
            if os.path.exists(image_path):
                st.image(image_path, caption="Processed Frame", use_column_width=True)
