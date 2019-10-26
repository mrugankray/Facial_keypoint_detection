import glob
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms ,datasets , utils 

key_pts_frame = pd.read_csv('/content/training_frames_keypoints.csv')
#print(len(key_pts_frame))

n = 0
img_name = key_pts_frame.iloc[n,0]
key_pts = key_pts_frame.iloc[n,1:].as_matrix()
key_pts = key_pts.astype('float').reshape(-1,2)
#key_pts = key_pts.numpy()
#key_pts = torch.from_numpy(key_pts)
#key_pts = key_pts.flatten()
'''
print(key_pts.size)
print('Image name: ', img_name)
print('Landmarks shape: ', key_pts.shape)
print(f'First 4 key pts: {key_pts}')

# print out some stats about the data
print('Number of images: ', key_pts_frame.shape)
print(key_pts)'''
#Number of images:  (3462, 137) NOTE: name of the image is included so column size is 137

def show_keypoints(img = img_name,  keypoints = key_pts):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(keypoints[:,0], keypoints[:,1], s = 20, marker = '.', c = 'm')
    plt.show()

#show_keypoints(mpimg.imread(os.path.join('/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/cvnd_exercises-master/p1_facial_keypoints-master/P1_Facial_Keypoints-master/data/training/', img_name)))

class FacialKeyPointDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx,0])
        img = mpimg.imread(img_name)

        if img.shape[2] == 4:
            img = img[:,:,0:3]

        key_pts = key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1,2)
        sample = {'image_name':self.key_pts_frame.iloc[idx,0], 'image': img, 'key_points': key_pts}

        if self.transform:
            sample = self.transform(sample)
        return sample

'''face_dataset = FacialKeyPointDataset(csv_file = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/cvnd_exercises-master/p1_facial_keypoints-master/P1_Facial_Keypoints-master/data/training_frames_keypoints.csv', root_dir = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/cvnd_exercises-master/p1_facial_keypoints-master/P1_Facial_Keypoints-master/data/training/')

print('length of dataset', len(face_dataset))
print(type(face_dataset))

num_to_disp = 3
for i in range(num_to_disp):
    #define the size of image
    fig = plt.figure(figsize=(20,10))

    #random select sample
    rand_i = np.random.randint(0, len(face_dataset))
    sample = face_dataset[rand_i]

    # print relevant info about the image
    print(f"i = {i}, image_name = {sample['image_name']}, image_shape = {sample['image'].shape}, key_points_shape = {sample['key_points'].shape}")

    show_keypoints(img = sample['image'], keypoints = sample['key_points'])'''



## Apllying Transformation ##



class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['key_points']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0
        #print('key_pts', key_pts)
        #print('key_pts_copy',key_pts_copy)


        return {'image': image_copy, 'key_points': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['key_points']

        #print('output size', self.output_size)

        h, w = image.shape[:2]
        #print('h, w', h, w)
        #print('key points', key_pts)
        if isinstance(self.output_size, int):
            if h > w:
                scale = float(w / self.output_size)
                new_h, new_w = h / scale, self.output_size
            else:
                scale = float(h / self.output_size)
                new_h, new_w = self.output_size, w / scale
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        #print('new_h, new_w', new_h, new_w)
        
        # scale the pts, too
        key_pts = key_pts / [scale, scale]
        #print('scaled key points', key_pts)

        return {'image': img, 'key_points': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['key_points']

        #print('random_crop_size', self.output_size)

        h, w = image.shape[:2]
        #print('h, w', h, w)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        #print('x, y', top, left)

        image = image[top: top + new_h,
                      left: left + new_w]

        #print('before changing key_pts', key_pts)

        key_pts = key_pts - [left, top]

        #print('after changing key_pts', key_pts)

        return {'image': image, 'key_points': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['key_points']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'key_points': torch.from_numpy(key_pts)}

# test out some of these transforms
'''rescale = Rescale(100)
crop = RandomCrop(50)
composed = transforms.Compose([Rescale(250),
                               RandomCrop(224)])

# apply the transforms to a sample image
test_num = 500
sample = face_dataset[test_num]

fig = plt.figure()
for i, tx in enumerate([rescale, crop, composed]):
    transformed_sample = tx(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tx).__name__)
    show_keypoints(transformed_sample['image'], transformed_sample['key_points'])

plt.show()

data_transfer = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])

transformed_dataset = FacialKeyPointDataset(csv_file = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/cvnd_exercises-master/p1_facial_keypoints-master/P1_Facial_Keypoints-master/data/training_frames_keypoints.csv', root_dir = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/cvnd_exercises-master/p1_facial_keypoints-master/P1_Facial_Keypoints-master/data/training/', transform = data_transfer)

# print some stats about the transformed data
print('Number of images: ', len(transformed_dataset['key_points']))

# make sure the sample tensors are the expected size
for i in range(1):
    sample = transformed_dataset[i]
    print(i, sample['image'].shape, sample['key_points'].shape)
'''