from flask import Flask, request, render_template, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import os
import uuid  # To generate unique image filenames

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load the trained model
model = torch.jit.load("plant_disease_cnn_scripted.pt")  # Use the scripted model
model.eval()

# **Get class names from dataset folders**
DATASET_PATH = "plant_data"  # Update if needed
class_names = sorted(os.listdir(DATASET_PATH))  # Folder names as class labels

# **Define upload folder**
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# **Serve the HTML Page**
@app.route("/")
def home():
    return render_template("index.html", prediction=None, image_url=None)

# **Serve Uploaded Images**
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# disease description
disease_descriptions = {
    "Apple - Alternaria Leaf Spot": "A fungal disease causing dark spots with concentric rings on apple leaves.",
    "Apple - Black Rot": "Caused by fungus, it leads to blackened fruit and leaf lesions.",
    "Apple - Brown Spot": "Fungal infection resulting in brown patches on leaves and fruit.",
    "Apple - Gray Spot": "A disease causing grayish moldy spots on apple leaves.",
    "Apple - Healthy": "No signs of disease — healthy apple leaf.",
    "Apple - Rust": "Bright orange spots on leaves caused by fungal rust pathogens.",
    "Apple - Scab": "Dark, scabby lesions caused by Venturia inaequalis fungus.",
    
    "Bell Pepper - Bacterial Spot": "Causes water-soaked spots that turn brown or black; common in humid conditions.",
    "Bell Pepper - Healthy": "No visible signs of disease or pest damage.",

    "Blueberry - Healthy": "Healthy blueberry foliage with no disease symptoms.",

    "Cassava - Bacterial Blight": "Serious disease causing wilting, spotting, and dieback of leaves.",
    "Cassava - Brown Streak Disease": "Causes streaks and root necrosis, leading to yield loss.",
    "Cassava - Green Mottle": "Leads to green mottling and distortion of cassava leaves.",
    "Cassava - Healthy": "No disease present — normal cassava leaf.",
    "Cassava - Mosaic Disease": "Virus-caused patterning and stunted growth of cassava.",

    "Cherry - Healthy": "No disease detected — healthy cherry leaf.",
    "Cherry - Powdery Mildew": "White, powdery fungal growth on leaf surfaces.",

    "Coffee - Healthy": "Leaf appears normal and disease-free.",
    "Coffee - Red Spider Mite": "Tiny mites that cause leaf discoloration and speckling.",
    "Coffee - Rust": "Causes yellow-orange rust spots, reducing yield and quality.",

    "Corn - Common Rust": "Reddish-brown pustules form on leaves, caused by Puccinia sorghi.",
    "Corn - Gray Leaf Spot": "Creates rectangular gray lesions on leaves, reducing photosynthesis.",
    "Corn - Healthy": "Corn plant with no signs of disease.",
    "Corn - Northern Leaf Blight": "Elongated gray-green lesions caused by Setosphaeria turcica.",

    "Grape - Black Measles": "A fungal disease that leads to black lesions and dried shoots.",
    "Grape - Black Rot": "Fungal infection causing black spots and fruit rot.",
    "Grape - Healthy": "Grape leaf free from any diseases.",
    "Grape - Leaf Blight": "Irregular brown spots on leaves, affecting photosynthesis.",

    "Orange - Citrus Greening": "Bacterial disease causing yellow shoots, bitter fruit, and dieback.",

    "Peach - Bacterial Spot": "Small, water-soaked lesions that become dark and sunken.",
    "Peach - Healthy": "No disease detected in peach foliage.",

    "Potato - Bacterial Wilt": "Causes wilting and browning due to Ralstonia bacteria.",
    "Potato - Early Blight": "Brown spots with concentric rings caused by Alternaria solani.",
    "Potato - Healthy": "Disease-free, healthy potato leaf.",
    "Potato - Late Blight": "Serious disease causing dark lesions and rapid rot.",
    "Potato - Leafroll Virus": "Virus that curls leaves upward and reduces yield.",
    "Potato - Mosaic Virus": "Mosaic pattern on leaves, often causing stunted growth.",
    "Potato - Nematode": "Microscopic worms that damage roots and affect plant health.",
    "Potato - Pests": "Insect pests damaging foliage and tubers.",
    "Potato - Phytophthora": "Fungal infection leading to root and stem rot.",
    "Potato - Spindle Tuber Viroid": "Stunted plants with narrow, upright leaves.",

    "Raspberry - Healthy": "No disease symptoms observed on raspberry leaf.",

    "Rice - Bacterial Blight": "Yellowing and drying of leaves due to bacterial infection.",
    "Rice - Blast": "Causes elliptical lesions and can severely reduce yield.",
    "Rice - Brown Spot": "Circular brown spots on leaves and grains.",
    "Rice - Tungro": "Virus spread by leafhoppers, stunting and discoloring rice plants.",

    "Rose - Healthy": "Healthy rose foliage without fungal or insect damage.",
    "Rose - Rust": "Rust-colored spores on the underside of leaves.",
    "Rose - Slug Sawfly": "Larvae that skeletonize leaves and create feeding holes.",

    "Soybean - Healthy": "Leaf is healthy with no disease or pest effects.",

    "Squash - Powdery Mildew": "White fungal growth on both sides of leaves.",

    "Strawberry - Healthy": "Healthy strawberry plant with no visible issues.",
    "Strawberry - Leaf Scorch": "Dark spots and leaf curling caused by fungus.",

    "Sugercane - Healthy": "No signs of disease in the sugarcane leaf.",
    "Sugercane - Mosaic": "Virus infection showing mottled leaf patterns.",
    "Sugercane - Red Rot": "Causes red discoloration and foul odor in stalks.",
    "Sugercane - Rust": "Orange pustules appear on leaves due to fungal infection.",
    "Sugercane - Yellow Leaf": "Yellowing of midribs, usually due to virus infection.",

    "Tomato - Bacterial Spot": "Spots that are dark and greasy-looking, reducing leaf area.",
    "Tomato - Early Blight": "Brown spots with concentric rings, commonly on older leaves.",
    "Tomato - Healthy": "No issues detected — healthy tomato leaf.",
    "Tomato - Late Blight": "Water-soaked lesions, often with white fungal growth on edges.",
    "Tomato - Leaf Curl": "Viral disease causing curling and stunting.",
    "Tomato - Leaf Mold": "Yellow spots on top, moldy growth underneath the leaf.",
    "Tomato - Mosaic Virus": "Causes mottled yellowing and malformed leaves.",
    "Tomato - Septoria Leaf Spot": "Small circular spots with dark borders and gray centers.",
    "Tomato - Spider Mites": "Tiny pests that cause speckled yellowing and webbing.",
    "Tomato - Target Spot": "Concentric ring spots on leaves and fruit.",
    
    "Watermelon - Anthracnose": "Dark, sunken spots on leaves and fruit caused by fungus.",
    "Watermelon - Downy Mildew": "Yellow angular spots with purplish mold on underside.",
    "Watermelon - Healthy": "No visible disease on the watermelon leaf.",
    "Watermelon - Mosaic Virus": "Mottled patterns on leaves, stunted growth."
}

# **Prediction Route**
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", prediction="No file uploaded!", image_url=None)

    file = request.files['file']

    # Ensure a file was selected
    if file.filename == '':
        return render_template("index.html", prediction="No file selected!", image_url=None)

    # Save and process the uploaded image
    filename = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Process the image for model prediction
    image = Image.open(file_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted_class_index = torch.argmax(output, 1).item()
        predicted_class_name = class_names[predicted_class_index]

    # ✅ Get description from dictionary
    description = disease_descriptions.get(predicted_class_name, "Description not available.")

    # ✅ Pass both prediction, description, and image path to the template
    return render_template("index.html", 
                           prediction=predicted_class_name,
                           description=description,
                           image_url=f"/uploads/{filename}")


if __name__ == '__main__':
    app.run(debug=True)
