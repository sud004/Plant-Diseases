import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the Keras model
model = tf.keras.models.load_model('model_1.h5')  # Update path if needed

def get_disease_info(predicted_class):
    disease_info = {
        'Pepper_bell__Bacterial_spot': {
            'Plant': 'Pepper Bell',
            'Issue': 'Bacterial Spot',
            'Description': 'Bacterial Spot in pepper bell leaves, caused by Xanthomonas bacteria.',
            'Remedy': 'Remove infected leaves, apply copper-based fungicides, and avoid overhead watering.'
        },
        'Pepper_bell__healthy': {
            'Plant': 'Pepper Bell',
            'Issue': 'Healthy',
            'Description': 'The pepper bell plant is healthy.',
            'Remedy': 'Maintain regular watering, provide balanced nutrients, and inspect leaves for any early signs of disease.'
        },
        'Potato__Early_blight': {
            'Plant': 'Potato',
            'Issue': 'Early Blight',
            'Description': 'Early blight in potato leaves, often caused by the fungus Alternaria solani.',
            'Remedy': 'Remove affected leaves, rotate crops, and apply fungicides as needed.'
        },
        'Potato__Late_blight': {
            'Plant': 'Potato',
            'Issue': 'Late Blight',
            'Description': 'Late blight in potato leaves, caused by Phytophthora infestans.',
            'Remedy': 'Use fungicides, remove infected plants, and avoid overhead watering to limit spread.'
        },
        'Potato__healthy': {
            'Plant': 'Potato',
            'Issue': 'Healthy',
            'Description': 'The potato plant is healthy.',
            'Remedy': 'Continue monitoring and provide regular care, such as balanced fertilization and proper watering.'
        },
        'Tomato__Bacterial_spot': {
            'Plant': 'Tomato',
            'Issue': 'Bacterial Spot',
            'Description': 'Bacterial spot affecting tomato leaves, caused by Xanthomonas bacteria.',
            'Remedy': 'Remove infected leaves, use disease-resistant varieties, and apply copper-based fungicides.'
        },
        'Tomato__Early_blight': {
            'Plant': 'Tomato',
            'Issue': 'Early Blight',
            'Description': 'Early blight in tomato leaves, caused by Alternaria solani.',
            'Remedy': 'Prune affected leaves, rotate crops, and apply appropriate fungicides.'
        },
        'Tomato__Late_blight': {
            'Plant': 'Tomato',
            'Issue': 'Late Blight',
            'Description': 'Late blight affecting tomato plants, caused by Phytophthora infestans.',
            'Remedy': 'Apply fungicides, remove infected plants, and avoid overhead watering.'
        },
        'Tomato__Leaf_Mold': {
            'Plant': 'Tomato',
            'Issue': 'Leaf Mold',
            'Description': 'Leaf mold in tomato plants, typically caused by the fungus Passalora fulva.',
            'Remedy': 'Provide adequate ventilation, remove infected leaves, and use fungicides if necessary.'
        },
        'Tomato__Septoria_leaf_spot': {
            'Plant': 'Tomato',
            'Issue': 'Septoria Leaf Spot',
            'Description': 'Septoria leaf spot caused by the fungus Septoria lycopersici, affecting tomato leaves.',
            'Remedy': 'Remove infected leaves, apply fungicides, and practice crop rotation.'
        },
        'Tomato__Spider_mites_Two_spotted_spider_mite': {
            'Plant': 'Tomato',
            'Issue': 'Spider Mites',
            'Description': 'Damage caused by two-spotted spider mites on tomato leaves.',
            'Remedy': 'Spray with insecticidal soap, maintain humidity, and remove heavily infested leaves.'
        },
        'Tomato__Target_Spot': {
            'Plant': 'Tomato',
            'Issue': 'Target Spot',
            'Description': 'Target spot disease, caused by Corynespora cassiicola, affecting tomato leaves.',
            'Remedy': 'Remove affected foliage, avoid wetting foliage, and apply fungicides if needed.'
        },
        'Tomato__Tomato_YellowLeaf_Curl_Virus': {
            'Plant': 'Tomato',
            'Issue': 'Yellow Leaf Curl Virus',
            'Description': 'Tomato Yellow Leaf Curl Virus, transmitted by whiteflies, causing leaf curl and yellowing.',
            'Remedy': 'Use disease-resistant varieties, control whiteflies, and remove infected plants.'
        },
        'Tomato__Tomato_mosaic_virus': {
            'Plant': 'Tomato',
            'Issue': 'Tomato Mosaic Virus',
            'Description': 'Infection caused by Tomato Mosaic Virus, leading to mottling and distortion of leaves.',
            'Remedy': 'Remove infected plants, disinfect tools, and avoid handling plants when wet.'
        },
        'Tomato__Healthy': {
            'Plant': 'Tomato',
            'Issue': 'Healthy',
            'Description': 'The tomato plant is healthy and free from disease.',
            'Remedy': 'Provide balanced nutrients, ensure proper watering, and monitor regularly for signs of disease.'
        },
        'Other': {
            'Plant': 'Unknown',
            'Issue': 'Unknown',
            'Description': 'The uploaded image does not match any known plant disease.',
            'Remedy': 'Ensure you are uploading an image related to plant diseases.'
    }
    }
    return disease_info.get(predicted_class, {
        'Plant': 'Unknown',
        'Issue': 'Unknown',
        'Description': 'No description available.',
        'Remedy': 'No remedy available.'
    })

# Class names
CLASS_NAME = [
    'Pepper_bell__Bacterial_spot', 'Pepper_bell__healthy',
    'Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy',
    'Tomato__Bacterial_spot', 'Tomato__Early_blight',
    'Tomato__Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot',
    'Tomato__Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Healthy','Other' 
]

# Streamlit App Interface
st.title('Plant Disease Classifier')
st.markdown("""
    This app classifies plant diseases based on uploaded leaf images. 
    Upload an image and let the model predict the plant condition.
""")

# File upload and camera capture
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with col2:
    camera_input = st.camera_input("Capture Image", label_visibility="collapsed")

img = None
if uploaded_file:
    img = Image.open(uploaded_file)
elif camera_input:
    img = Image.open(io.BytesIO(camera_input.getvalue()))

if img:
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image for the model
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    confidence = round(100 * np.max(predictions[0]), 2)

    # Confidence threshold
    confidence_threshold = 60.0
    if confidence < confidence_threshold:
        predicted_class = "Other"
    else:
        predicted_class = CLASS_NAME[np.argmax(predictions[0])]

    # Get disease information
    disease_info = get_disease_info(predicted_class)

    # Display prediction results
    st.subheader("Prediction Results:")
    st.write(f"**Plant Name:** {disease_info['Plant']}")
    st.write(f"**Issue:** {disease_info['Issue']}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Description:** {disease_info['Description']}")
    st.write(f"**Remedy:** {disease_info['Remedy']}")