from deepface import DeepFace

def predict_emotion(img_path):

    result = DeepFace.analyze(
        img_path=img_path,
        actions=["emotion"],
        enforce_detection=False
    )

    emotion = result[0]["dominant_emotion"]

    return emotion


if __name__ == "__main__":

    img = input("Enter image path: ")

    emotion = predict_emotion(img)

    print("Emotion :", emotion)