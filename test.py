import cv2
import torch 
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np

# defining Network

# defining Network

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, stride = 1)
        self.conv2 = nn.Conv2d(32, 64, 2, stride = 1)
        self.conv3 = nn.Conv2d(64, 128, 2,stride = 1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(15488,1000)
        self.fc2 = nn.Linear(1000, 136)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.5)
    def forward(self, x):
        
        x = self.dropout1(self.maxpool(F.relu(self.conv1(x))))   
        x = self.dropout2(self.maxpool(F.relu(self.conv2(x))))
        x = self.dropout3(self.maxpool(F.relu(self.conv3(x))))
        x = x.view(-1,15488)
        x = self.dropout4(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
model = Net()
print(model)
state_dict = torch.load('/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/face_keypoints_det/model/new_arch/e37/model.pth')
model.load_state_dict(state_dict)
model = model.double()

#model = model.type(torch.DoubleTensor)
with torch.no_grad():
    model.eval()
    #model.eval() generates points in the lower right side of the image so i had to comment it
    face_pic = cv2.imread('/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/face_keypoints_det/modi_cropped.jpg')

    face_pic_copy = face_pic.copy()

    face_pic_copy = cv2.cvtColor(face_pic_copy, cv2.COLOR_BGR2GRAY)

    # Normalizing
    face_pic_copy = face_pic_copy/255

    face_pic_copy = cv2.resize(face_pic_copy, (96,96), interpolation = cv2.INTER_AREA)
    print('face_pic_copy.shape before reshaping', face_pic_copy.shape)

    # reshape the image in numpy readable form
    face_pic_copy = face_pic_copy.reshape(face_pic_copy.shape[0], face_pic_copy.shape[1], 1)
    print('reshaping face_pic_copy to numpy readableform', face_pic_copy.shape)

    # convert it into a tensor 
    #face_pic_copy = np.transpose(face_pic_copy, (2,0,1))
    face_pic_copy = face_pic_copy.transpose((2,0,1))
    face_pic_copy_tensor = torch.from_numpy(face_pic_copy)
    print('face_pic_copy_tensor shape', face_pic_copy_tensor.shape)
    #print('face_pic_copy_tensor', face_pic_copy_tensor[0])

    # batch size will be 1
    face_pic_copy_tensor = face_pic_copy_tensor.unsqueeze(0)
    print('face_pic_copy_tensor after unsqueeze', face_pic_copy_tensor.shape)

    #face_pic_copy_tensor = Variable(face_pic_copy_tensor)

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print('device: ',device)

    if device == 'cuda':
        face_pic_copy_tensor = face_pic_copy_tensor.type(torch.cuda.FloatTensor)
        face_pic_copy_tensor.cuda()
        model.cuda()

    # scores
    logits = model(face_pic_copy_tensor)
    logits = logits.view(logits.size()[0], 68, -1)
    print('logits: ',logits)
    pred_key_pts = logits[0].data
    print('pred_key_pts(unnormalised): ', pred_key_pts)

    if device == 'cuda':
        pred_key_pts = pred_key_pts.cpu()
    pred_key_pts = pred_key_pts.numpy()

    # normalize the pred_key_pts
    pred_key_pts = pred_key_pts*50.0 + 100

    print('pred_key_pts(normalised): ', pred_key_pts)

    plt.imshow(np.squeeze(face_pic_copy_tensor), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.scatter(pred_key_pts[:,0], pred_key_pts[:,1], c='m', s=20,marker='.')
    plt.savefig('/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/face_keypoints_det/pred_img13.png')
    #plt.imshow('face_pic_copy', face__copypic)


# finding kernels in the model
weights = model.conv1.weight.data
w = weights.numpy()
print(w[1][0])
plt.imshow(w[1][0], cmap='gray')
plt.show()

# convolute with another image
for i in range(16):
    plt.subplot(4, 4, i+1)
    filtered_img = cv2.filter2D(face_pic, -1, w[i][0])
    plt.imshow(filtered_img)
    plt.xticks([])
    plt.yticks([])
plt.savefig('/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/face_keypoints_det/filtered_img.png')
plt.show()