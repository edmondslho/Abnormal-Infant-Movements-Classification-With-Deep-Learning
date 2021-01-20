import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim

import torch.utils.data as data

from model import FCNet, Conv1DNet, Conv2DNet

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################
# This is the implementation of the Abnormal Infant Movements Classification Framework proposed in
# [1] McCay et al., "Abnormal Infant Movements Classification with Deep Learning on Pose-based Features", IEEE Access, vol. 8, pp. 51582-51592, 2020.
#
# The HOJO2D and HOJD2D features are extracted using the methods proposed in
# [2] McCays et al., "Establishing Pose Based Features Using Histograms for the Detection of Abnormal Infant Movements", IEEE EMBC, pp. 5469-5472, July 2019.
#
# GitHub page: https://github.com/edmondslho/SMARTBabies
#
# Please feel free to contact the corresponding author Edmond S. L. Ho (e.ho@northumbria.ac.uk or edmond@edho.net) for any questions and comments
#
# Please cite these papers in your publications if it helps your research:
#
# @ARTICLE{McCay:DeepBaby,
#   author={K. D. {McCay} and E. S. L. {Ho} and H. P. H. {Shum} and G. {Fehringer} and C. {Marcroft} and N. D. {Embleton}},
#   journal={IEEE Access},
#   title={Abnormal Infant Movements Classification With Deep Learning on Pose-Based Features},
#   year={2020},
#   volume={8},
#   pages={51582-51592},
#   doi={10.1109/ACCESS.2020.2980269}
# }
#
# @INPROCEEDINGS{McCay:PoseBaby,
#    author={K. D. {McCay} and E. S. L. {Ho} and C. {Marcroft} and N. D. {Embleton}},
#    booktitle={2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
#    title={Establishing Pose Based Features Using Histograms for the Detection of Abnormal Infant Movements},
#    year={2019},
#    pages={5469-5472},
#    doi={10.1109/EMBC.2019.8857680}
# }
#
###############################################################################
#
# An example on training and testing a classifier with the settings as follows:
feature_set = 'Limbs8+VelLimbs8'
bins = 8

#feature_set = 'Limbs16+VelLimbs16'
#bins = 16

# select a model to be used
modelStr = 'FCNet'      # other models are available, such as Conv1DNet-1, Conv1DNet-2, Conv2DNet-1, and Conv2DNet-2

# addiional settings
batch_size = 3
dropoutRate = 0.1
epochs = 50
print_every = 25
learningRate = 0.0005

# Note: Due to the random initialization of our newly proposed deep learning
# frameworks, the performance of the classifier may vary in different trials.
# We reported the best performance of classifiers in the paper [1]
###############################################################################


print("1. Loading data - {}".format(feature_set))
trainV = pd.read_csv("data/{}_X.csv".format(feature_set),dtype = np.float32,header=None)
trainL = pd.read_csv("data/{}_Y.csv".format(feature_set),dtype = np.float32,header=None)

# total number of samples in the MINI-RGBD dataset
ttlVid = 12
allAccuracy = np.zeros(ttlVid)

# leave-one-out cross-validation
for a in range(ttlVid):
    print('{}_{}_dropout_{} | split data - set {}'.format(feature_set, modelStr, dropoutRate, a))

    # define train/test split
    trainID = list(range(ttlVid))
    testID = [ a, a ]
    trainID.remove(a)

    features_numpy = trainV.values[trainID]
    targets_numpy = trainL.values[trainID]

    test_features_numpy = trainV.values[testID]
    test_targets_numpy = trainL.values[testID]

    # create feature and targets tensor for the training set
    featuresTrain = torch.from_numpy(features_numpy)
    targetsTrain = torch.from_numpy(targets_numpy).type(torch.LongTensor)

    # create feature and targets tensor for the testing set
    featuresTest = torch.from_numpy(test_features_numpy)
    targetsTest = torch.from_numpy(test_targets_numpy).type(torch.LongTensor)

    #print('2. Pytorch train and test sets')
    train = torch.utils.data.TensorDataset(featuresTrain.to(device),targetsTrain.to(device))
    splittest = torch.utils.data.TensorDataset(featuresTest.to(device),targetsTest.to(device))

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)#False)
    test_loader = torch.utils.data.DataLoader(splittest, batch_size = batch_size, shuffle = False)


    #print('3. Defining DL Model')

    #print('Implement a function for the validation pass')
    def validation(model, testloader, criterion):
        test_loss = 0
        accuracy = 0
        for images, labels in testloader:
            output = model.forward(images)
            labels = labels.squeeze_().to(device)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

            return test_loss, accuracy


    input_size = trainV.shape[1]
    output_size = 2     # binary classification

    # our proposed models
    if modelStr=="FCNet":
        hidden_sizes = [240, 60, 15, 4]
        model = FCNet(input_size, output_size, hidden_sizes, drop_p=dropoutRate);
    if modelStr=="Conv1DNet-1":
        hidden_sizes = [1, 16, 16]
        model = Conv1DNet(input_size, output_size, hidden_sizes, drop_p=dropoutRate, version=1);
    if modelStr=="Conv1DNet-2":
        hidden_sizes = [1, 16, 32]
        model = Conv1DNet(input_size, output_size, hidden_sizes, drop_p=dropoutRate, version=2);
    if modelStr=="Conv2DNet-1":
        hidden_sizes = [1, 4, 4]
        model = Conv2DNet(input_size, output_size, hidden_sizes, drop_p=dropoutRate, version=1, binSize=bins);
    if modelStr=="Conv2DNet-2":
        hidden_sizes = [1, 4, 8]
        model = Conv2DNet(input_size, output_size, hidden_sizes, drop_p=dropoutRate, version=2, binSize=bins);

    # switch to GPU (if available)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    #print('4. Build DL Model')
    steps = 0

    # store loss and iteration
    loss_list = []
    iteration_list = []
    accuracy_list = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in iter(train_loader):
            steps += 1

            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)

            labels = labels.squeeze_()

            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if e % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, test_loader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
                      "Test Accuracy: {:.3f} (#{})".format(accuracy/len(test_loader), len(test_loader)))

                # store the best performance
                acc = accuracy / len(test_loader)
                if acc > allAccuracy[a]:
                    allAccuracy[a] = acc

                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(steps)
                accuracy_list.append(accuracy/len(test_loader))
                running_loss = 0

                # Make sure training is back on
                model.train()

print(allAccuracy)
print("Avg: {:.3f}".format(np.average(allAccuracy)))

# write all results to a file
np.savetxt("allResults_{}_{}_dropout_{}_batchSize_{}.csv".format(feature_set, modelStr, dropoutRate, batch_size), allAccuracy, delimiter=",")