import os
import tempfile
import streamlit as st
from PIL import Image

from predict_all import predict_all
from dress_color import detect_dress_color

st.set_page_config(
    page_title="Nationality Detection System",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Nationality Detection System")

uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    temp_path = temp.name
    temp.close()

    image.save(temp_path)

    try:

        nationality, age, emotion = predict_all(temp_path)

        st.subheader("Prediction Results")

        st.success(f"Nationality : {nationality}")

        if nationality == "Indian":

            color = detect_dress_color(temp_path)

            st.write(f"**Estimated Age :** {age}")
            st.write(f"**Emotion :** {emotion}")
            st.write(f"**Dress Color :** {color}")

        elif nationality == "United States":

            st.write(f"**Estimated Age :** {age}")
            st.write(f"**Emotion :** {emotion}")

        elif nationality == "African":

            color = detect_dress_color(temp_path)

            st.write(f"**Emotion :** {emotion}")
            st.write(f"**Dress Color :** {color}")

        else:

            st.write(f"**Emotion :** {emotion}")

    except Exception as e:

        st.error(str(e))

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)