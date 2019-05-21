import MyTrainer
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

# Fixed configuration section
dataset = 'D2'
num_layers = [2, 2]
input_dimensions = 4
num_hidden_nodes = [200, 100]
init_weights = 0.01
p_ipm = 0.5
num_iter_per_decay = 100
datadir = 'data/'
train_file_name = 'gen_data.test.npz'
test_file_name = 'gen_data.train.npz'
outdir = 'results/'
experiments = 20
iterations = 200000
batch_size = 100
validation_size = 0.3

# Grid search parameters
dropout_range = np.linspace(0.5, 0.75, num=5)
my_dropout_list = []
for e1 in dropout_range:
    for e2 in dropout_range:
        my_dropout_list.append([e1, e2])
my_alpha_list = np.linspace(0.001, 0.01, num=10)
my_lambda_list = np.linspace(0.001, 0.01, num=10)
my_l_rate_list = np.linspace(0.001, 0.01, num=10)
my_l_rate_decay_list = np.linspace(0.8, 1, num=10)

param_grid = {
    'dropout': my_dropout_list,
    'alpha': my_alpha_list,
    'lambda': my_lambda_list,
    'l_rate': my_l_rate_list,
    'l_rate_decay': my_l_rate_decay_list
}
combinations = list(ParameterGrid(param_grid))

dropout_list = []
alpha_list = []
lambda_list = []
l_rate_list = []
l_rate_decay_list = []
train_f_error_list = []
train_cf_error_list = []
val_f_error_list = []
val_cf_error_list = []
ate_list = []

i = 0
for combination in combinations:
    print('Evaluating ' + str(i) + ' out of ' + str(len(combinations)) + ' combinations...')
    result = MyTrainer.TrainCFRNet(num_layers, input_dimensions, num_hidden_nodes, init_weights, combination['dropout'],
                                   p_ipm, combination['alpha'], combination['lambda'], combination['l_rate'],
                                   combination['l_rate_decay'], num_iter_per_decay, datadir, train_file_name,
                                   test_file_name, outdir, experiments, iterations, batch_size, validation_size)
    dropout_list.append(str(combination['dropout']))
    alpha_list.append(str(combination['alpha']))
    lambda_list.append(str(combination['lambda']))
    l_rate_list.append(str(combination['l_rate']))
    l_rate_decay_list.append(str(combination['l_rate_decay']))
    train_f_error_list.append(result.avg_train_f_error)
    train_cf_error_list.append(result.avg_train_cf_error)
    val_f_error_list.append(result.avg_val_f_error)
    val_cf_error_list.append(result.avg_val_cf_error)
    ate_list.append(result.ate)
    i += 1
myDf = pd.DataFrame()
myDf['Dropout'] = dropout_list
myDf['Alpha'] = alpha_list
myDf['Lambda'] = lambda_list
myDf['Learning rate'] = l_rate_list
myDf['Learning rate decay'] = l_rate_decay_list
myDf['Train F error'] = train_f_error_list
myDf['Train CF error'] = train_cf_error_list
myDf['Val F error'] = val_f_error_list
myDf['Val CF error'] = val_cf_error_list
myDf['ATE'] = ate_list
myDf.to_csv('Result' + dataset + '.csv', index=False)
print('Done')
