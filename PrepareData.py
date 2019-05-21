import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

experiments = 20

df = pd.read_csv('D2.csv')
train, test = train_test_split(df, test_size=0.3)

splittedTrain = np.array_split(train, experiments)
splittedTest = np.array_split(test, experiments)

# Prepare train data
xTrainList = []
yfTrainList = []
ycfTrainList = []
tTrainList = []

for trainSplit in splittedTrain:
    x = trainSplit[['bought_sim', 'bought_same', 'clicked', 'searched']].values
    yf = trainSplit[['buy']].values
    ycf = trainSplit[['buy_CF']].values
    t = trainSplit[['adv']].values
    xTrainList.append(x)
    yfTrainList.append(yf)
    ycfTrainList.append(ycf)
    tTrainList.append(t)

xTrain = np.dstack(xTrainList)
yfTrain = np.concatenate(yfTrainList, axis=1)
ycfTrain = np.concatenate(ycfTrainList, axis=1)
tTrain = np.concatenate(tTrainList, axis=1)
np.savez('gen_data.train.npz', x=xTrain, yf=yfTrain, ycf=ycfTrain, t=tTrain)

# Prepare test data
xTestList = []
yfTestList = []
ycfTestList = []
tTestList = []

for testSplit in splittedTest:
    x = testSplit[['bought_sim', 'bought_same', 'clicked', 'searched']].values
    yf = testSplit[['buy']].values
    ycf = testSplit[['buy_CF']].values
    t = testSplit[['adv']].values
    xTestList.append(x)
    yfTestList.append(yf)
    ycfTestList.append(ycf)
    tTestList.append(t)

xTest = np.dstack(xTestList)
yfTest = np.concatenate(yfTestList, axis=1)
ycfTest = np.concatenate(ycfTestList, axis=1)
tTest = np.concatenate(tTestList, axis=1)
np.savez('gen_data.test.npz', x=xTest, yf=yfTest, ycf=ycfTest, t=tTest)

print('Done')
