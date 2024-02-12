import streamlit
import numpy as np
from PIL import Image
import pickle

# 'model' and 'test_df' are defined and loaded
with open('gender_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit App
st.title("Gender Prediction App")

# File uploader for user input image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to grayscale, resize, and preprocess using Pillow
    processed_image = image.convert('L')  # Convert to grayscale
    processed_image = processed_image.resize((48, 48))
    processed_image = np.array(processed_image).reshape(48, 48, 1)

    # Display the processed image
    st.image(processed_image, caption="Processed Image", use_column_width=True)

    # Make predictions
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    gender_label = "Male" if prediction < 0.5 else "Female"

    # Display prediction result
    st.subheader("Prediction Result:")
    st.write(f"Predicted gender: {gender_label}")
    st.write(f"Prediction score: {prediction[0][0]}")

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(np.stack(test_df['image'].values), test_df['gender_label'])
    st.subheader("Model Evaluation:")
    st.write(f'Test accuracy: {test_acc}')
