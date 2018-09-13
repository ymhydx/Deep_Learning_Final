import package
import os
import numpy as np

def get_num(x):
    temp1=x.split('.')
    temp2=temp1[0].split()
    return int(temp2[1])

def get_num_v2(x):
    temp1=x.split('.')
    temp2=temp1[0].split('_')
    try:
        signal=int(temp2[1])
    except:
        signal=0
    temp3=temp2[0].split()
    return int(temp3[1]),signal
#############################################
folders = os.listdir('./vgg19')
indices={}
for folder in folders:
    temp=[]
    num_files=len(os.listdir('./vgg19' + '/' + folder))
    train_num=int(num_files*0.6)
    valid_num=int((num_files-train_num)/2)
    seq=np.random.permutation(np.array(range(num_files)))
    temp.append(seq[:train_num])
    temp.append(seq[train_num:train_num+valid_num])
    temp.append(seq[train_num+valid_num:])
    indices[folder]=temp

#############################################
#kept frames
num_frames=70

dataset = []
labels = []
one_hot = package.one_hot_encoder(51)
folders = os.listdir('./vgg19')

cls = 0
for folder in folders:
    files = os.listdir('./vgg19' + '/' + folder)
    for file in files:
        key=False
        for i in indices[folder][0]:
            if get_num(file)==i:
                key=True
                break                         
        if key:
            features = np.load('./vgg19' + '/' + folder + '/' + file)
            features = package.subsample(features,num_frames)
            dataset.append(features)
            labels.append(np.array(one_hot[cls]))
    cls += 1

dataset = np.array(dataset)
labels = np.array(labels)

np.save('./train_features.npy', dataset)
np.save('./train_labels.npy', labels)

#############################################
#kept frames
num_frames=70

dataset = []
labels = []
one_hot = package.one_hot_encoder(51)
folders = os.listdir('./vgg19')

cls = 0
for folder in folders:
    files = os.listdir('./vgg19' + '/' + folder)
    for file in files:
        key=False
        for i in indices[folder][1]:
            if get_num(file)==i:
                key=True
                break                         
        if key:
            features = np.load('./vgg19' + '/' + folder + '/' + file)
            features = package.subsample(features,num_frames)
            dataset.append(features)
            labels.append(np.array(one_hot[cls]))
    cls += 1

dataset = np.array(dataset)
labels = np.array(labels)

np.save('./validation_features.npy', dataset)
np.save('./validation_labels.npy', labels)
#############################################
#kept frames
num_frames=70

dataset = []
labels = []
one_hot = package.one_hot_encoder(51)
folders = os.listdir('./vgg19')

cls = 0
for folder in folders:
    files = os.listdir('./vgg19' + '/' + folder)
    for file in files:
        key=False
        for i in indices[folder][2]:
            if get_num(file)==i:
                key=True
                break                         
        if key:
            features = np.load('./vgg19' + '/' + folder + '/' + file)
            features = package.subsample(features,num_frames)
            dataset.append(features)
            labels.append(np.array(one_hot[cls]))
    cls += 1

dataset = np.array(dataset)
labels = np.array(labels)

np.save('./test_features.npy', dataset)
np.save('./test_labels.npy', labels)
#############################################
#kept frames
num_frames=70

dataset = []
labels = []
one_hot = package.one_hot_encoder(51)
folders = os.listdir('./vgg19_augmented')

cls = 0
for folder in folders:
    files = os.listdir('./vgg19_augmented' + '/' + folder)
    for file in files:
        num1,num2=get_num_v2(file)
        key1=False
        key2=False
        for i in indices[folder][0]:
            if num1==i:
                key1=True
                break
        if num2==0 or num2==10 or num2==20:
            key2=True                  
        if key1 and key2:
            features = np.load('./vgg19_augmented' + '/' + folder + '/' + file)
            features = package.subsample(features,num_frames)
            dataset.append(features)
            labels.append(np.array(one_hot[cls]))
    cls += 1

dataset = np.array(dataset)
labels = np.array(labels)

np.save('./train_features_aug.npy', dataset)
np.save('./train_labels_aug.npy', labels)