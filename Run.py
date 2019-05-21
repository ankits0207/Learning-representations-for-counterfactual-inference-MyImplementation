import MyTrainer

# Configuration section
num_layers = [2, 2]
input_dimensions = 4
num_hidden_nodes = [200, 100]
init_weights = 0.01
dropout = [1.0, 1.0]
p_ipm = 0.5
my_alpha = 0.001
my_lambda = 0.001
l_rate = 0.001
l_rate_decay = 0.9
num_iter_per_decay = 100
datadir = 'data/'
train_file_name = 'gen_data.test.npz'
test_file_name = 'gen_data.train.npz'
outdir = 'results/'
experiments = 2
iterations = 3000
batch_size = 100
validation_size = 0.3

test_run = MyTrainer.TrainCFRNet(num_layers, input_dimensions, num_hidden_nodes, init_weights, dropout, p_ipm,
                                 my_alpha, my_lambda, l_rate, l_rate_decay, num_iter_per_decay, datadir, train_file_name,
                                 test_file_name, outdir, experiments, iterations, batch_size, validation_size)