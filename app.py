import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Force TensorFlow to use CPU (Fixes CUDA error on Render)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses unnecessary warnings

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = load_model("model.h5", compile=False)  # Load without compiling
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # Explicitly compile
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Define class labels
class_labels = ["pituitary", "glioma", "notumor", "meningioma"]  # Multi-class classification

# Define the uploads folder
UPLOAD_FOLDER = "./uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = (128, 128)  # Adjust if needed
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions)

    return class_labels[predicted_class_index], confidence_score

# Route for main page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            file_location = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)

            return render_template(
                "index.html",
                result=result,
                confidence=f"{confidence * 100:.2f}%",
                file_path=f"/uploads/{file.filename}",
            )

    return render_template("index.html", result=None)

# Route to serve uploaded images
@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Ensure correct port for Render deployment
    app.run(host="0.0.0.0", port=port, debug=False)
