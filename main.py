import os
import pathlib
import shutil
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152, ResNet152_Weights
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained resnet model
model = resnet152(weights=ResNet152_Weights.DEFAULT)
model.eval()

# Preprocessing for images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(image_path, device):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    model.to(device)
    
    with torch.no_grad():
        features = model(img_tensor)
    
    features = features.flatten().cpu().numpy()
    return features

# Function to compute similarity scores between query image and database images
def compute_similarity(query_features, database_features):
    similarity_scores = cosine_similarity(query_features.reshape(1, -1), database_features)
    return similarity_scores.flatten()

# Directory containing database images
database_dir = '/mnt/d/show/ai models/ziana/copy'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract features for all images in the database
database_features = []
database_image_paths = []
for filename in os.listdir(database_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(database_dir, filename)
        database_image_paths.append(image_path)
        features = extract_features(image_path, device=device)
        database_features.append(features)

# database_features = np.array(database_features)

target_dir = "/mnt/d/show/ai models/ziana/ready"
group_name_prefix = "group"
counter = 0
threshold = 0.7

# create a group list and add the picked image to the list
# loop through rest of the images in the database
# compare the picked image and the current image
# if the similarity is above the threshold put the current image in the group and remove from the database
# create a directory in the target directory for the group and copy each image to the directory
# while there are images in the database
while len(database_image_paths) > 1:
# pick the top image
    group = []
    group.append(database_image_paths.pop())
    referenceImageFeature = database_features.pop()
    similarity_scores = compute_similarity(referenceImageFeature, np.array(database_features))
    print(len(similarity_scores), len(database_features), len(database_image_paths))
    added_index = []
    for i in range(len(similarity_scores)):
        if similarity_scores[i] >= threshold:
            group.append(database_image_paths[i])
            added_index.append(i)
    dir_name = f"{group_name_prefix}_{counter}"
    dir_path =os.path.join(target_dir, dir_name) 
    if not os.path.exists(dir_path):
        os.mkdir(dir_path, 0o755)
    for image in group:
        target_path = os.path.join(dir_path, pathlib.Path(image).name)
        print(target_path)
        print(image)
        shutil.copy(image, target_path)
    counter+=1

    database_features_new = []
    database_image_paths_new = []
    for i, image in enumerate(database_image_paths):
        if i not in added_index:
            database_image_paths_new.append(image) 
        
    for i, feature in enumerate(database_features):
        if i not in added_index:
            database_features_new.append(feature) 
            
    database_features = database_features_new
    database_image_paths = database_image_paths_new
