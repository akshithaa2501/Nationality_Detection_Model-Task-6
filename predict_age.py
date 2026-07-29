from deepface import DeepFace

def predict_age(img_path):
    result = DeepFace.analyze(
        img_path=img_path,
        actions=["age"],
        enforce_detection=False
    )

    return int(result[0]["age"])


if __name__ == "__main__":
    img = input("Enter image path: ")
    age = predict_age(img)
    print("Estimated Age:", age)