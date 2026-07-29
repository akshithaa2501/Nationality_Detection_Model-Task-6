from deepface import DeepFace

def predict_nationality(img_path):

    result = DeepFace.analyze(
        img_path=img_path,
        actions=["race"],
        enforce_detection=False
    )

    race = result[0]["dominant_race"].lower()

    if race == "indian":
        return "Indian"

    elif race == "black":
        return "African"

    elif race == "white":
        return "United States"

    else:
        return "Others"


if __name__ == "__main__":

    img = input("Enter image path: ")

    nationality = predict_nationality(img)

    print("Nationality :", nationality)