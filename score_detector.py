"""A highly optimized OCR detector for scores in training mode of FIFA.

Usage:
    1. Call prediction function from other program:

        from score_detector import ScoreDetector
        detector = ScoreDetector()
        score_screen = ... # Get the screen-shot of score
        result, success = detector.predict(model, score_screen)

    2. Use <generate_training_data> function to generate more training samples.

    3. Use <train_model> function to train a new model.
"""
import cv2
import os
import time
import numpy as np
from matplotlib import pyplot as plt
# from utils.grab_screen import get_screenshot, get_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as vision_func

save_dir = os.path.join('score_detector_model', 'raw_data')
model_path = os.path.join('score_detector_model', 'ocr_model_180723.pth')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.display_shape = False
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 14 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        if self.display_shape: print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        if self.display_shape: print(x.shape)
        x = x.view(-1, 16 * 14 * 2)
        if self.display_shape: print(x.shape)
        x = F.relu(self.fc1(x))
        if self.display_shape: print(x.shape)
        x = F.relu(self.fc2(x))
        if self.display_shape: print(x.shape)
        x = self.fc3(x)
        return x
    
    def transforms(self):
        return transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def split_reward_screen(reward_screen, crop_right=True, verbose=0):
    # Threshold to black and white
    reward_screen = cv2.cvtColor(reward_screen, cv2.COLOR_RGB2GRAY)
    reward_screen = 255 - reward_screen
    reward_screen = cv2.adaptiveThreshold(reward_screen, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)

    # Split into characters by histogram
    # Generate vertical histogram
    kernel = np.ones((1, 1), np.uint8)
    split_img = cv2.resize(reward_screen, None, fx=2, fy=2)
    split_img = cv2.erode(split_img, kernel, iterations=1)
    split_img = (255 - split_img) / 255
    # Crop the left size
    # split_img = split_img[:, 15:]
    if crop_right:
        split_img = split_img[:, :-15]
    hist = np.sum(split_img, axis=0)
    # Parameters while segmenting
    # Relative values, should be fine with different source image's shape
    up_thres = 8 * split_img.shape[0] / 70
    down_thres = 8 * split_img.shape[0] / 70
    up_thres_delta = 1 * split_img.shape[0] / 70
    down_thres_delta = 1 * split_img.shape[0] / 70
    hori_thres = 10 * split_img.shape[1] / 105
    # Find the turning point in the histogram
    up_points, down_points = [], []
    state = 0
    for i in range(1, hist.shape[0]):
        # Judge whether an upward/downward point by:
        # 1. Last state
        # 2. Absolute value
        # 3. Acceleration
        # 4. Distance to previous upward point
        # 5. (Only for downward points) Distance to previous upward point
        is_up = (state == 0)
        is_up = is_up and (hist[i-1] < up_thres)
        is_up = is_up and (hist[i] - hist[i-1] > up_thres_delta)
        is_up = is_up and (len(up_points) == 0 or i > (up_points[-1] + hori_thres))
        if is_up:
            state = 1
            up_points.append(i)
            continue
        is_down = (state == 1)
        is_down = is_down and (hist[i-1] < down_thres)
        is_down = is_down and (hist[i] - hist[i-1] < down_thres_delta)
        is_down = is_down and (len(down_points) == 0 or i > (down_points[-1] + hori_thres))
        is_down = is_down and (len(up_points) == 0 or i > (up_points[-1] + hori_thres))
        if is_down:
            state = 0
            down_points.append(i)
    
    # Visualize
    if verbose:
        plt.clf()
        plt.subplot(325)
        for i in up_points:
            plt.plot(np.ones(50) * i, range(50), 'g')
        for i in down_points:
            plt.plot(np.ones(50) * i, range(50), 'b')
        print('Hist shape:', hist.shape)
        plt.plot(hist)

        # plt.show()

        # cv2.imshow('score', split_img)
        # cv2.waitKey(0)

    digits = []
    # Check whether segmentation success
    if len(up_points) == len(down_points):
        # Actually segment characters
        resize_x = 20 # int(20 * split_img.shape[1] / 105)
        resize_y = 70 # split_img.shape[0]
        # print('Reshape to {}-{} (x-y)'.format(resize_x, resize_y))
        
        for i in range(len(up_points)):
            digit = split_img[:, up_points[i]:down_points[i]]
            digit = cv2.resize(digit, (resize_x, resize_y))
            digits.append(digit)
            
        if verbose:
            for idx, digit in enumerate(digits):
                id_subplot = 320 + idx + 1
                plt.subplot(id_subplot)
                plt.imshow(digit)
            
            plt.show()

    return digits

'''no use'''
def generate_training_data():
    # Check output folder
    for i in range(10):
        dir_path = os.path.join(save_dir, str(i))
        if not os.path.exists(dir_path):
            print('Creating folder:', dir_path)
            os.mkdir(dir_path)

    # The reward meter at top right corner of game screen
    reward_screen = get_score()
    # cv2.imshow("screen", reward_screen)
    # cv2.waitKey()

    # Split the digits by histogram
    digits = split_reward_screen(reward_screen, verbose=0)

    # For each digit
    if len(digits) > 0:
        # Ask for human labeling
        for i in range(len(digits)):
            digit_im = (255 * digits[i]).astype(np.uint8)
            cv2.imshow('digit', digit_im)
            key = cv2.waitKey(0)
            label = key - 48
            assert 0 <= label <= 9, 'Invalid label {} (0 <= label <= 9)'.format(label)
            print(label)
            filename = os.path.join(str(label), str(int(time.time()*1e3)) + '.png')
            cv2.imwrite(os.path.join(save_dir, filename), digit_im)
        plt.close('all')
    else:
        print('No digits found.')

'''no use'''
def train_model(filename=model_path):
    epochs = 20
    classes = [str(i) for i in range(10)]
    
    net = Net()
    transform = net.transforms()

    # Data loader
    trainset = torchvision.datasets.ImageFolder(root=save_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=1)

    # testset = torchvision.datasets.ImageFolder(root='./OCR_training', transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                         shuffle=False, num_workers=2)

    # Visualize, get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    # plt.show()
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

    torch.save(net.state_dict(), filename)
    print('Finished Training, model saved to', filename)

    # Test accuracy 
    correct = 0
    total = 0
    with torch.no_grad():
        # for data in testloader:
        for data in trainloader:
            images, labels = data
            outputs = net(images)
            predicted = np.argmax(outputs.data, axis=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the training images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in trainloader:
        images, labels = data
        outputs = net(images)
        predicted = np.argmax(outputs.data, axis=1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


class ScoreDetector():
    def __init__(self, filename=model_path):
        self.net = Net()
        self.net.load_state_dict(torch.load(filename))

    def predict(self, reward_screen=None, crop_right=True, is_test=False):
        transforms = self.net.transforms()

        if is_test:
            # The reward meter at top right corner of game screen
            reward_screen = get_score()
            # cv2.imshow("screen", reward_screen)
            # cv2.waitKey()

        # Split the digits by histogram
        digits = split_reward_screen(reward_screen, crop_right=crop_right, verbose=0)
        
        # For each digit
        labels = ''
        if len(digits) > 0:
            for i in range(len(digits)):
                # Preprocess
                digit_im = (255 * digits[i]).astype(np.uint8)
                digit_im = np.stack([digit_im for i in range(3)], axis=2)
                transformed = transforms(vision_func.to_pil_image(digit_im)).unsqueeze(0)
                prediction = self.net(transformed)
                label = torch.argmax(prediction)
                labels += str(label.numpy())
                # cv2.imshow('digit', digit_im)
                # cv2.waitKey(0)
            labels = int(labels)
            success = True
        else:
            # print('No digits found.')
            labels = 0
            success = False
        
        return labels, success

if __name__ == '__main__':
    # generate_training_data()
    # train_model()

    detector = ScoreDetector()
    
    last_score = 0
    while 1:
        score, success = detector.predict(is_test=True)
        if not success:
            print('ERROR', end='\r')
        elif score != last_score:
            print(score)
            last_score = score