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

# Configuration parameters
splits = [0.7, 0.15, 0.15]  # train, val, test set size
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


# Data splitting

# this dictionary creates an id for each label
label2idx = {label: i for i, label in enumerate(label_names)}

# we store the filenames and labels in lists
files, targets = [], []
for label in label_names:
    label_files = glob(os.path.join(data_dir, label, "*.jpg"))
    files += label_files
    targets += [label2idx[label]] * len(label_files)

# we split the lists into training, validation, and test
train_files, temp_files, train_labels, temp_labels = train_test_split(
    files, targets, train_size=splits[0], stratify=targets, random_state=seed)

val_size = splits[1] / (splits[1] + splits[2])  # proportion of val in temp
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, train_size=val_size,
    stratify=temp_labels, random_state=seed)
