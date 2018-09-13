# import numpy as np
# import package

# indices=np.random.permutation(np.array(range(set.shape[0])))

# features = np.load('./features.npy')
# labels = np.load('./labels.npy')
# train_features, validation_features, test_features, train_labels, validation_labels, test_labels = package.split(features, labels)
# np.save('./train_features.npy',train_features)
# np.save('./validation_features.npy',validation_features)
# np.save('./test_features.npy',test_features)
# np.save('./train_labels.npy',train_labels)
# np.save('./validation_labels.npy',validation_labels)
# np.save('./test_labels.npy',test_labels)
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

import os
temp=os.listdir('./vgg19_augmented/brush_hair')
for i in range(len(temp)):
    temp[i]=get_num_v2(temp[i])
print(temp)