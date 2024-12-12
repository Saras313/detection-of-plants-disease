import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from graph import plot_disease_graph

# Disease classes and descriptions
disease_classes = {
    'Brownspot': "Brownspot is a bacterial disease that leads to dark, necrotic spots on rice leaves.",
    'Bacterialblight': "Bacterial blight is caused by bacteria that result in water-soaked lesions on the rice leaves.",
    'No Disease': "No disease detected in the image."
}

# Load ResNet-50 and modify the final layer to match your number of classes (38)
model = models.resnet50(weights=None)  # Do not download new weights
model.fc = nn.Sequential(nn.Linear(in_features=model.fc.in_features, out_features=38))

# Load your custom trained weights (no need to download new ones)
state_dict = torch.load('./model/trained_rn50_model.pth', map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Placeholder for device
device = torch.device("cpu")
model.to(device)

def predict_disease(image):
    """Function to predict disease based on uploaded image."""
    # Convert RGBA to RGB (if the image has 4 channels)
    if image.shape[2] == 4:
        image = image[:, :, :3]  # Discard the alpha channel and keep the RGB channels

    # Preprocess the image to match model's input shape
    img_array = cv2.resize(image, (224, 224))  # Resize image to match model input
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Convert to torch tensor and move to device
    img_tensor = torch.tensor(img_array).permute(0, 3, 1, 2).float().to(device)  # Convert to correct tensor format

    # Model inference
    with torch.no_grad():
        logits = model(img_tensor)
        prediction = F.softmax(logits, dim=1)
        predicted_class_index = prediction.argmax(dim=1).item()  # Get predicted class index
        predicted_class = list(disease_classes.keys())[predicted_class_index]

    return predicted_class


def upload_and_predict():
    """Handles image upload and prediction"""
    st.title('Rice Plant Disease Detection')
    st.markdown("""## Upload images of the rice plant to detect any diseases.""")

    # Add option to select single or multiple image uploads
    upload_type = st.radio("Choose upload type:", options=["Single Image", "Multiple Images"], key="upload_type")

    if upload_type == "Single Image":
        uploaded_image = st.file_uploader("Choose an image of the rice plant", type=["jpg", "png", "jpeg"], key="single_image")

        if uploaded_image is not None:
            # Open and display the image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Convert the image to an array for processing
            image_array = np.array(image)

            # Predict disease
            disease = predict_disease(image_array)

            # Display the result and disease description
            st.subheader(f"Detected Disease: {disease}")
            st.write(disease_classes[disease])
            st.info("Please consult a specialist for treatment options.")

    elif upload_type == "Multiple Images":
        uploaded_images = st.file_uploader("Choose multiple images of rice plants", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="multiple_images")

        if uploaded_images:
            predictions = {}

            for uploaded_image in uploaded_images:
                # Open the image
                image = Image.open(uploaded_image)
                st.image(image, caption=f"Uploaded Image: {uploaded_image.name}", use_container_width=True)

                # Convert the image to an array
                image_array = np.array(image)

                # Predict disease
                disease = predict_disease(image_array)

                # Add prediction to results
                predictions[uploaded_image.name] = disease

            # Display predictions in a graph
            st.write("Generating Disease Prediction Graph...")
            plot_disease_graph(predictions)
