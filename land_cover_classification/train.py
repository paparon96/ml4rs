import os
import random
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights

from tqdm import tqdm

# Ensuring reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

data_dir = './data/EuroSAT_RGB/'
label_names = sorted(os.listdir(data_dir))
if '.DS_Store' in label_names:
    label_names.remove('.DS_Store')

plt.figure(figsize=(15, 10))
for i, label in enumerate(label_names):
    filename = os.listdir(os.path.join(data_dir, label))[0]
    path = os.path.join(data_dir, label, filename)
    img = Image.open(path)
    plt.subplot(4, 5, i + 1)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')
    
plt.savefig('./images/landcover-examples.jpg', dpi=300)
plt.tight_layout()
plt.show()


# Displaying RGB channels of a sample image
label = label_names[3]
path = os.path.join(data_dir, label, os.listdir(os.path.join(data_dir, label))[0])
img = Image.open(path).convert("RGB")
channels = img.split()
titles = ["Original", "Red", "Green", "Blue"]
cmaps = [None, "Reds", "Greens", "Blues"]

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(img if i == 0 else channels[i - 1], cmap=cmaps[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('./images/landcover-rgb.jpg', dpi=300)
plt.show()