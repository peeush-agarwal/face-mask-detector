import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse

def loaders(dataset_path, validation_size = 0.2, batch_size = 4, verbose=False):
    '''
    Input: dataset_path = Path to the dataset directory
    Output: Loaders for train and validation images
    '''
    transformations = transforms.Compose([transforms.Resize((300,400)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(dataset_path, transformations)
    n_items = len(dataset)
    valid_set_size = int(n_items*validation_size)
    train_set_size = n_items - valid_set_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size])

    if verbose:
        print(len(train_set))
        print(len(valid_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader

def imshow(loader, classes):
    images, labels = next(iter(loader))
    images = torchvision.utils.make_grid(images)

    plt.imshow(images.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(','.join([classes[i.item()] for i in labels]))
    plt.show()

def load_model(classes, verbose = False):
    model = torchvision.models.vgg16_bn(pretrained=True)

    for params in model.parameters():
        params.requires_grad = False

    if verbose:
        print('Original model:')
        print(model)

    num_features = model.classifier[6].in_features
    layers = list(model.classifier.children())[:-1]
    layers.extend([nn.Linear(num_features, len(classes))])
    model.classifier = nn.Sequential(*layers)

    if verbose:
        print('Updated model:')
        print(model)
    return model

def train(model, criterion, optim, train_loader, epochs, lr, batch_size, display_plots = True, verbose = False):
    model.train()
    losses = []
    accuracies = []
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        for batch_id, data in enumerate(train_loader):
            images, labels = data

            optim.zero_grad()

            out = model(images)
            _, preds = torch.max(out, dim=1)
            loss = criterion(out, labels)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            corrects = torch.sum(preds == labels.data)
            running_corrects += corrects
            if verbose:
                print(f'Batch:{batch_id}, Loss:{loss.item()}, Acc:{corrects}/{batch_size}')
        print(f'Epoch:{epoch}, Loss:{running_loss/len(train_loader)}, Acc:{running_corrects/len(train_loader)}')
        losses.append(running_loss/len(train_loader))
        accuracies.append(running_corrects/len(train_loader))
    
    if display_plots:
        plt.plot(losses, 'r-')
        plt.plot(accuracies, 'g+')
        plt.title('Loss vs Accuracy')
        plt.show()

def infer(model, valid_loader, itr = None, verbose = False):
    model.eval()
    for i, data in enumerate(valid_loader):
        if itr is not None and i >= itr:
            break
        
        images, labels = data
        predictions = model(images)
        if verbose:
            print('Predictions:')
            print(predictions)
        _, predictions = torch.max(predictions, dim=1)
        
        print(f'Final Predictions:{predictions}')
        print(f'Ground truth     :{labels}')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", help="Path to dataset directory", required=True)
    ap.add_argument("-m", "--model-path", help="Path to save model", default="mask_detector.pth")
    args = ap.parse_args()

    print(f"Path to dataset: {args.dataset}")
    print(f"Save model to path: {args.model_path}")

    BATCH_SIZE = 16
    VALID_SIZE = 0.2
    EPOCHS = 1
    LR = 0.0005

    classes = ['with_mask', 'without_mask']


    train_loader, valid_loader = loaders(args.dataset, VALID_SIZE, BATCH_SIZE)
    imshow(train_loader, classes)
    # model = load_model()
    # criterion = nn.CrossEntropyLoss()
    # optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # train(model, criterion, optim, train_loader, EPOCHS, LR, BATCH_SIZE)

    # infer(model, valid_loader, itr = 1, verbose=True)

