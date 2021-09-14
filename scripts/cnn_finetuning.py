import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import Image
import json
import pandas as pd


def minibatch(filename):

    img = Image.open(filename).convert('RGB')
    itensor = preprocess(img)
    ibatch = itensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        ibatch = ibatch.to('cuda')
        model.to('cuda')

    return(ibatch)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    #img_dir = '/home/pablo/Documents/phd/project1/stimuli/Kar_repo/images/' #where the images are
    #img_files = glob.glob(img_dir + "*.png")

    imagenet_dir = '/home/pablo/Documents/phd/project1/imagenet/'
    imagenet_classes = imagenet_dir + 'imagenet_class_index.json'

    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

    #fix weights to prevent retraining in original layers
    for param in model.parameters():
        param.requires_grad = False

    #modify fc8 to generate an output of 10 features
    model.classifier._modules['6'] = nn.Linear(4096, 10) #last linear layer in the classifier

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])

    data_dir = '/home/pablo/Documents/phd/project1/dnn_dataset/kar_version/all'
    out_dir = '/home/pablo/Documents/phd/project1/analysis/kar_version/tables/'
    image_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
    dataset_files = image_dataset.imgs

    #train_dataset, val_dataset = torch.utils.data.random_split(image_dataset, (880, 440))
    #datasets = {'train': train_dataset, 'val': val_dataset}

    nnets = 100
    data = pd.DataFrame()
    for i in range(nnets):

        dataset_ixs = list(range(len(image_dataset)))
        np.random.shuffle(dataset_ixs)

        train_idx, val_idx = dataset_ixs[int(len(dataset_ixs)/3):], dataset_ixs[:int(len(dataset_ixs)/3)]

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        samplers = {'train': train_sampler, 'val': val_sampler}

        dataloaders = {x: torch.utils.data.DataLoader(image_dataset, batch_size=1,
                                                     shuffle=False, sampler=samplers[x])
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
        class_names = image_dataset.classes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                           num_epochs=20)
        #save model
    #    torch.save(model.state_dict(), data_dir + 'alexnet_ft_kar')
        torch.save(model, data_dir + 'alexnet_ft_kar_n' + str(i))
        #load model
    #    model = torch.load(data_dir + 'alexnet_ft_kar')
    #    model.eval()

############################TEST#####################


        testfiles = [dataset_files[x][0] for x in val_idx]
        test_ids = [x[len(data_dir)+1:-4].split('/') for x in testfiles]

        batch = [minibatch(x) for x in testfiles]

        model.eval()
        with torch.no_grad():
            output = [model(x) for x in batch]

        nout = [torch.nn.functional.softmax(x[0], dim=0) for x in output]
        nout = [x.numpy() for x in nout]
        ans = [x.argmax() for x in nout]
        pc = [x.max() for x in nout]

        ids = [test_ids[x] + [testfiles[x]] for x in range(len(testfiles))]
        df1 = pd.DataFrame(ids, columns = ['obj', 'img','file'])
        df2 = pd.DataFrame(nout)
        df3 = pd.DataFrame(ans, columns = ['ans'])
        dat = pd.concat([df1,df2,df3],axis=1)

        dat['pc'] = pc
        dat['correct'] = [int(dat.obj[x]) == dat.ans[x] for x in range(len(dat))]
        dat['nnet'] = i

        data = data.append(dat)

    data.to_csv(out_dir + 'alexnet_inference_iter.csv')
