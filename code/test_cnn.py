import numpy as np
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize, Normalize
import torch
import torch.nn as nn
from Animal_class import AnimalDataset
from resnet_model import ResNet
import argparse
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(description="Test a Animal CNN model")
    parser.add_argument("--model", '-m', type=str, default='tensorboard/animals/epoch_best.pt')
    parser.add_argument("--image", '-i', type=str, default='../test_img/1.jpg')
    args = parser.parse_args()
    return args


# Transformer:
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])
root = '/home/trile/Desktop/ML_DL/DEEP_LEARNING/Exercise/demo/'
dataset = AnimalDataset(root=root, train=True, transform=transform)
classes = dataset.categories
print(classes)
softmax = torch.nn.Softmax(dim=1)

if __name__ == '__main__':
    args = get_args()

    # SET UP MODEL, FOLDER AND PARAMETER BEFORE TRAINING AND TESTING PROCESS

    # Set device is CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = ResNet(num_classes=10).to(device)
    # get checkpoint:
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Get test img and convert to Tensor
    img = Image.open(args.image).convert('RGB')
    img.show()
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)

    # predict the image according to model
    with torch.no_grad():
        outputs = model(img)
        probabilities = softmax(outputs)
    idx = torch.argmax(probabilities)
    print("The image is about {}".format(classes[idx]))
