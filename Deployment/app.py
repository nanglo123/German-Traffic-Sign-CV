### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch


from timeit import default_timer as timer
from typing import Tuple, Dict
from create_model import create_CNN
import torchvision
from torchvision import transforms
# Setup class names
with open("class_names.txt", "r") as f: # reading them in from class_names.txt
    class_names = [food_name.strip() for food_name in  f.readlines()]
    
### 2. Model and transforms preparation ###    

# Create model
model = create_CNN()

# Load saved weights
model.load_state_dict(
    torch.load(
        f="model_adam_epoch25.pth",
        map_location=torch.device("cpu"),  # load to CPU because gpu not guaranteed
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    t = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.ToTensor()
        ])

    # Transform the target image and add a batch dimension
    img = t(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(model(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "GTSRB - German Traffic Sign Recognition by Tenzin Nanglo"
description = "CNN created for the GTSRB Dataset, achieved 98% test accuracy"
article = "Created a 4 block CNN consisting of three convolutional blocks and one classifier block. The model was trained on 25 epochs using an Adam Optimizer with a learning rate of .001. The design of the CNN architecture was inspired by other projects conducted on this dataset. There were no additional preprocessing procedures done on the data besides resizing them into 32x32 images."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create Gradio interface 
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article,
)

# Launch the app!
demo.launch()
