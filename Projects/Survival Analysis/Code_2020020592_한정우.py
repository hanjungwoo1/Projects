import pandas as pd
import os
import glob
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, Input, Activation, Conv1D
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split


use_gpu = False


def data_loader ():

    #data set path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    Data = os.path.join(os.path.dirname(BASE_DIR), "Survival_Analysis/Data")
    Files = glob.glob(os.path.join(Data, "*"))

    #data set attribute
    data = pd.read_excel(Files[0],usecols=[22,23,24,25,26,27,28,29,30,
                                           #31,32, #add
                                           #34,35,36,37,38,39,40,41,42,43,44,45,46,47,48, #add
                                           52,53,54,55,56,57,58,
                                           60,61,
                                           #63, #add
                                           64,65,66,67,68,69,70,71,
                                           #72,73,74 #add
                         ])

    #data label
    label = pd.read_excel(Files[0],usecols=[114])
    label = label.to_numpy()

    #data to numpy
    data = data.to_numpy()
    data[data == 0] = 'nan' # 0 to 'nan'

    #check data no, dim
    no, dim = data.shape
    print("no : " + str(no))
    print("dim : " + str(dim))

    # mapping indication data
    indi_data = np.where(data>0,0,data)
    indi_data = np.where(data<0,0,indi_data)
    indi_data = np.nan_to_num(indi_data, nan=1)

    # imputation for train
    data = np.nan_to_num(data, nan=1)

    # # Normalization (0 to 1)(vertical)
    Min_Val = np.zeros(dim)
    Max_Val = np.zeros(dim)

    for i in range(dim):
        Min_Val[i] = np.min(data[:, i])
        data[:, i] = data[:, i] - np.min(data[:, i])
        Max_Val[i] = np.max(data[:, i])
        data[:, i] = data[:, i] / (np.max(data[:, i]))

    return data, indi_data, label


miss_data,  indi_data, label = data_loader()

data = miss_data
check_data = data
l_label = label


#====================================================
# Start Gain Model **Step1

no, dim = miss_data.shape

mb_size = 288
p_miss = 0.5
p_hint = 0.9
alpha = 100
train_rate = 0.8

H_Dim1 = dim
H_Dim2 = dim


# indexing for Train, Test
idx = np.random.permutation(no)
Train_No = int(no * train_rate)
Test_No = no - Train_No


# Train / Test Features
trainX = miss_data[idx[:Train_No],:]
testX = miss_data[idx[Train_No:],:]

# Train / Test Missing Indicators
trainM = indi_data[idx[:Train_No],:]
testM = indi_data[idx[Train_No:],:]

#xavier initailizing
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)

# sampling, inidication
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A > p
    C = 1.*B
    return C



#%% 1. Discriminator
    # Output is multi-variate

D_W1 = torch.tensor(xavier_init([dim*2, H_Dim1]),requires_grad=True)     # Data + Hint as inputs
D_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True)

D_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True)
D_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True)

D_W3 = torch.tensor(xavier_init([H_Dim2, dim]),requires_grad=True)
D_b3 = torch.tensor(np.zeros(shape = [dim]),requires_grad=True)       # Output is multi-variate

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


#%% 2. Generator

G_W1 = torch.tensor(xavier_init([dim*2, H_Dim1]),requires_grad=True)     # Data + Mask as inputs (Random Noises are in Missing Components)
G_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True)

G_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True)
G_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True)

G_W3 = torch.tensor(xavier_init([H_Dim2, dim]),requires_grad=True)
G_b3 = torch.tensor(np.zeros(shape = [dim]),requires_grad=True)

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]


# %% 1. Generator
def generator(new_x, m):
    inputs = torch.cat(dim=1, tensors=[new_x, m])  # Mask + Data Concatenate
    G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
    G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)
    G_prob = torch.sigmoid(torch.matmul(G_h2, G_W3) + G_b3)  # [0,1] normalized Output

    return G_prob


# %% 2. Discriminator
def discriminator(new_x, h):
    inputs = torch.cat(dim=1, tensors=[new_x, h])  # Hint + Data Concatenate
    D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)
    D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
    D_logit = torch.matmul(D_h2, D_W3) + D_b3
    D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output

    return D_prob


# %% 3. Other functions
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size=[m, n])


# Mini-batch generation , But not use
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def discriminator_loss(M, New_X, H):
    # Generator
    G_sample = generator(New_X, M)
    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_New_X, H)

    # %% Loss
    D_loss = -torch.mean(M * torch.log(D_prob + 1e-5) + (1 - M) * torch.log(1. - D_prob + 1e-5))
    return D_loss


def generator_loss(X, M, New_X, H):
    # %% Structure
    # Generator
    G_sample = generator(New_X, M)

    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_New_X, H)

    # %% Loss
    G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + 1e-5))
    MSE_train_loss = torch.mean((M * New_X - M * G_sample) ** 2) / torch.mean(M)

    G_loss = G_loss1 + alpha * MSE_train_loss

    # %% MSE Performance metric
    MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
    return G_loss, MSE_train_loss, MSE_test_loss


def test_loss(X, M, New_X):
    # %% Structure
    # Generator
    G_sample = generator(New_X, M)

    # %% MSE Performance metric
    MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
    return MSE_test_loss, G_sample


optimizer_D = torch.optim.Adam(params=theta_D)
optimizer_G = torch.optim.Adam(params=theta_G)

print("====start====")

for it in tqdm(range(1000)):

    # %% Inputs
    X_mb = trainX   #missing data

    Z_mb = sample_Z(mb_size, dim)  #indicator
    M_mb = trainM
    H_mb1 = sample_M(mb_size, dim, 1 - p_hint)
    H_mb = M_mb * H_mb1

    New_X_mb = X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

    if use_gpu is True:
        X_mb = torch.tensor(X_mb, device="cuda")
        M_mb = torch.tensor(M_mb, device="cuda")
        H_mb = torch.tensor(H_mb, device="cuda")
        New_X_mb = torch.tensor(New_X_mb, device="cuda")
    else:
        X_mb = torch.tensor(X_mb)
        M_mb = torch.tensor(M_mb)
        H_mb = torch.tensor(H_mb)
        New_X_mb = torch.tensor(New_X_mb)

    optimizer_D.zero_grad()
    D_loss_curr = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
    D_loss_curr.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()
    G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
    G_loss_curr.backward()
    optimizer_G.step()

    # %% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))
        print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
        print()


Z_mb = sample_Z(360, dim)
M_mb = indi_data
X_mb = miss_data

New_X_mb = X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

if use_gpu is True:
    X_mb = torch.tensor(X_mb, device='cuda')
    M_mb = torch.tensor(M_mb, device='cuda')
    New_X_mb = torch.tensor(New_X_mb, device='cuda')
else:
    X_mb = torch.tensor(X_mb)
    M_mb = torch.tensor(M_mb)
    New_X_mb = torch.tensor(New_X_mb)

MSE_final, Sample = test_loss(X=X_mb, M=M_mb, New_X=New_X_mb)

print('Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))

imputed_data = X_mb + (1-M_mb) * Sample
print("Imputed test data:")
# np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

if use_gpu is True:
    print(imputed_data.cpu().detach().numpy())
    data = imputed_data.cpu().detach().numpy()
else:
    print(imputed_data.detach().numpy())
    data = imputed_data.detach().numpy()


#====================================================
# Start Linear regression-based classification Model **Step2


num_classes = 14
batch_size = 360
epochs = 10000

data = np.array(data, dtype= float)
print(data.shape)

c= []
for a in label:
    b = str(a)
    c.append(b[3:6])

label = np.array(c, dtype= str)


#mapping label for classification
label[label=='006'] = 0
label[label=='012'] = 1
label[label=='018'] = 2
label[label=='024'] = 3
label[label=='036'] = 4
label[label=='048'] = 5
label[label=='060'] = 6
label[label=='072'] = 7
label[label=='084'] = 8
label[label=='096'] = 9
label[label=='108'] = 10
label[label=='120'] = 11
label[label=='132'] = 12
label[label=='144'] = 13
print(label)

label = keras.utils.to_categorical(label, num_classes)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, shuffle=False)

#model Sequential
model = Sequential()
model.add(Dense(20, input_dim=dim, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(14, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(data, label, batch_size=batch_size, epochs=epochs, verbose=1)

score = model.evaluate(data, label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


print(imputed_data)
print(check_data)



#====================================================
# Start KaplanMeier **Step3


c= []
for a in l_label:
    b = str(a)
    c.append(b[3:6])

label = np.array(c, dtype= int)


from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt


test0 = data[:,10] #Ventricles
test1 = data[:,11] #Hippocampus
test2 = data[:,12] #WholeBrain
test3 = data[:,13] #Entorhinal

ax = plt.subplot(111)
kmf = KaplanMeierFitter()
kmf.fit(test0, label, label = 'Ventricles')
kmf.plot(ax=ax)

kmf.fit(test1, label, label = 'Hippocampus')
kmf.plot(ax=ax)

kmf.fit(test2, label, label = 'WholeBrain')
kmf.plot(ax=ax)

kmf.fit(test3, label, label = 'Entorhinal')



plot = kmf.plot_survival_function()
plot.set_xlabel('time (144month)')
plot.set_ylabel('survival function, $\hat{S}(t)$')
plt.show()




