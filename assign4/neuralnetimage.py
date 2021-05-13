import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

# Change pretrained from false-true to pretrain or vice-versa
alexnet = models.alexnet(pretrained=True)

# Transform for image to fit into alexnet
transform = transforms.Compose([            
 transforms.Resize(256),                    
 transforms.CenterCrop(224),               
 transforms.ToTensor(),                    
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 )])

img = Image.open("WelshCorgi.jpeg")

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Uncomment to show the neuralnet details
# print(alexnet)

# Code to run Alexnet in eval
alexnet.eval()
output = alexnet(batch_t)

#  Get the classes from the imagenet file to be used
with open('classes.txt') as f:
  labels = [line.strip() for line in f.readlines()]

_, indices = torch.sort(output, descending=True)
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]