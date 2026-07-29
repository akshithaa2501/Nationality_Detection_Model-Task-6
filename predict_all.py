from deepface import DeepFace

def predict_all(img_path):

    result = DeepFace.analyze(
        img_path=img_path,
        actions=["race", "age", "emotion"],
        enforce_detection=False
    )

    result = result[0]

    # ---------- Nationality ----------
    race = result["dominant_race"].lower()

    if race == "indian":
        nationality = "Indian"

    elif race == "white":
        nationality = "United States"

    elif race == "black":
        nationality = "African"

    else:
        nationality = "Others"

    # ---------- Age ----------
    age = int(result["age"])

    # ---------- Emotion ----------
    emotion = result["dominant_emotion"].capitalize()

    return nationality, age, emotion


if __name__ == "__main__":

    img = input("Enter image path : ")

    nationality, age, emotion = predict_all(img)

    print("Nationality :", nationality)
    print("Estimated Age :", age)
    print("Emotion :", emotion)