import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler 
from torchvision import models, transforms
from transform_data.face_keypoint_det import FacialKeyPointDataset, Normalize, Rescale, RandomCrop, ToTensor
import numpy as np
# dataset #
data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])

transformed_train_dataset = FacialKeyPointDataset(csv_file = '/content/training_frames_keypoints.csv', root_dir = '/content/training/', transform = data_transform)

transformed_test_dataset = FacialKeyPointDataset(csv_file = '/content/test_frames_keypoints.csv', root_dir = '/content/test/', transform = data_transform)

print('length of dataset', len(transformed_train_dataset))
print(type(transformed_train_dataset))

# training indices used for validation #
len_dataset = len(transformed_train_dataset)
indices = list(range(len_dataset))
np.random.shuffle(indices)
split = int(np.floor(0.2*len_dataset))
train_idx, test_idx = indices[split:], indices[:split]

# trn_sampler #
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(test_idx)

# training loader #
num_workers = 0
batch_size = 64

train_loader = DataLoader(transformed_train_dataset, num_workers = num_workers, batch_size = batch_size , sampler = train_sampler)

val_loader = DataLoader(transformed_train_dataset, num_workers = num_workers, batch_size = batch_size, sampler = val_sampler)

test_loader = DataLoader(transformed_test_dataset, num_workers = num_workers, batch_size = batch_size , shuffle = True)

# checking if gpu is available #
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# defining model architecture
model = models.vgg11_bn(pretrained=False)

# redesigning the classifier part to output 136 features and the convolution part to accept grayscale image
from collections import OrderedDict
model.features[0] = nn.Conv2d(1,64, 3, stride=(1,2), padding=(1,1))
print(model.features[0])
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 4096)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.5)),
    ('fc2', nn.Linear(4096, 4096)),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(0.5)),
    ('fc3', nn.Linear(4096, 1000)),
    ('relu3', nn.ReLU()),
    ('dropout3', nn.Dropout(0.5)),
    ('fc4', nn.Linear(1000, 136))
]))
model.classifier = classifier
print(model)

# defining loss funcion #
criterion = nn.CrossEntropyLoss

# defining optimizer #
optimizer = optim.SGD(model.parameters(), lr = 0.01)

if device == 'cuda':
    model = model.cuda()

print(device)

epochs = 50

trn_loss_list = []
val_loss_list = []
test_loss_list = []
val_min_loss = np.Inf

for epoch in range(epochs):
    trn_loss = 0
    test_loss = 0
    val_loss = 0

    trn_running_loss = 0
    test_running_loss = 0
    val_running_loss = 0

    for trn_i, trn_sample in enumerate(train_loader):

        # move all images and labels to the gpu if the system has a gpu
        trn_img = trn_sample['image']
        trn_key_pts = trn_sample['key_points']
        # flatten key points
        trn_key_pts = trn_key_pts.view(trn_key_pts.shape[0], -1)
        if device == 'cuda':
            trn_key_pts = trn_key_pts.type(torch.cuda.FloatTensor)
            trn_img = trn_img.type(torch.cuda.FloatTensor)
            trn_img = trn_img.to(device)
            trn_key_pts = trn_key_pts.to(device)

        # sets optimizer to zero grad
        optimizer.zero_grad()

        # log probability
        trn_log_ps = model(trn_img)

        # computing loss
        loss_trn = criterion(trn_log_ps, trn_key_pts)

        # backward propagation
        loss_trn.backward()

        # optimize the loss
        optimizer.step()

        # adding training loss at each image
        trn_running_loss += loss_trn.item()

    else:
        with torch.no_grad:
            # setting model to evaluation mode so that dropout does not ocur while we train the model
            model.eval()

            for val_i, val_sample in enumerate(val_loader):
                val_img = val_sample['images']
                val_label = val_sample['key_points']
                # flatten key points
                val_label = val_label.view(val_label.shape[0], -1)
                if device == 'cuda':
                    val_label = val_label.type(torch.cuda.FloatTensor)
                    val_img = val_img.type(torch.cuda.FloatTensor)
                    val_img = val_img.to(device)
                    val_label = val_label.to(device)

                loss_val = criterion(val_img, val_label)

                val_running_loss += loss_val.item()

            for test_i, test_sample in test_loader:
                test_img = test_sample['images']
                test_label = test_sample['key_points']
                # flatten key points
                test_label = test_label.view(test_label.shape[0], -1)
                if device == 'cuda':
                    test_label = test_label.type(torch.cuda.FloatTensor)
                    test_img = test_img.type(torch.cuda.FloatTensor)
                    test_img = test_img.to(device)
                    test_label = test_label.to(device)
                
                loss_test = criterion(test_img, test_label)

                test_running_loss += loss_test.item()

        trn_loss = trn_running_loss / len(train_loader)
        val_loss = val_running_loss / len(val_loader)
        test_loss = test_running_loss / len(test_loader)

        trn_loss_list = trn_loss_list.append(trn_loss)
        val_loss_list  = val_loss_list.append(val_loss)
        test_loss_list = test_loss_list.append(test_loss)

        print(f'epochs: {i+epoch} / {epochs}, training_loss: {trn_loss}, val_loss: {val_loss}, test_loss: {test_loss}')

        # setting model to training mode 
        model.train()

        if val_loss <= val_min_loss:
            print(f'validation loss decreased {val_loss} ---> {val_min_loss}. Saving model...')
            torch.save(model.state_dict(), 'model.pth')
            val_min_loss = val_loss

