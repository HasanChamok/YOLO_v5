from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load the pre-trained object detection model
model = torch.load("best.pt", map_location=torch.device('cpu'))
model.eval()

# Define a function to perform object detection on the input image
def perform_object_detection(image):
    # Process the image using your object detection model here
    # Replace this line with the code to perform object detection
    # and return the results as a text format
    # For example, you can use torchvision's object detection functions
    
    # Sample code to convert image to text format
    # For demonstration purposes, it converts the image to grayscale
    img = image.convert("L")
    img_text = "This is a placeholder text representing the detected objects."
    
    return img_text

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(BytesIO(image_data))
    
    # Perform object detection on the input image
    detection_result = perform_object_detection(image)
    
    # Save the detection result to a text file
    with open("detection_result.txt", "w") as text_file:
        text_file.write(detection_result)
    
    return {"message": "Object detection completed. Detection results saved to detection_result.txt."}
