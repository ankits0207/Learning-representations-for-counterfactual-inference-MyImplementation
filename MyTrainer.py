import os
import random
import datetime
import MyUtility
import MyNeuralNet
import numpy as np
import tensorflow as tf


class TrainCFRNet:
    def __init__(self, num_layers, input_dimensions, num_hidden_nodes, init_weights, dropout, p_ipm, my_alpha,
                 my_lambda, l_rate, l_rate_decay, num_iter_per_decay, datadir, train_file_name, test_file_name, outdir,
                 experiments, iterations, batch_size, validation_size):
        # Load the data
        train = MyUtility.Utility.load_data(datadir + train_file_name)
        test = MyUtility.Utility.load_data(datadir + test_file_name)

        # Creating directory to save the results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = outdir+timestamp+'/'
        os.mkdir(outdir)

        # Create tensorflow session object
        my_session = tf.Session()

        # Initialize input placeholders
        x = tf.placeholder("float", shape=[None, train['dim']], name='x')
        t = tf.placeholder("float", shape=[None, 1], name='t')
        y = tf.placeholder("float", shape=[None, 1], name='y')

        # Initialize parameter placeholders
        p_alpha = tf.placeholder("float", name='p_alpha')
        p_lambda = tf.placeholder("float", name='p_lambda')
        p_dropout_layer_set_1 = tf.placeholder("float", name='p_dropout_layer_set_1')
        p_dropout_layer_set_2 = tf.placeholder("float", name='p_dropout_layer_set_2')
        p_treatment_prob = tf.placeholder("float", name='p_treatment_prob')

        # Create parameters object and link parameter placeholders to MyNeuralNet
        params = MyUtility.Parameter(num_layers, input_dimensions, num_hidden_nodes, init_weights, p_ipm)
        # Create graph object
        my_nn = MyNeuralNet.MyNeuralNet(x, t, y, p_alpha, p_lambda, p_dropout_layer_set_1, p_dropout_layer_set_2,
                                        p_treatment_prob, params)

        # Setup optimizer
        my_step = tf.Variable(0, trainable=False)
        my_lr = tf.train.exponential_decay(l_rate, my_step, num_iter_per_decay,
                                        l_rate_decay, staircase=True)
        my_optimizer = tf.train.AdamOptimizer(my_lr)
        train_step = my_optimizer.minimize(my_nn.tot_loss, global_step=my_step)

        # Train/Val split
        row_count = train['x'].shape[0]
        valid_row_count = int(validation_size*row_count)
        train_row_count = row_count - valid_row_count
        indices = np.random.permutation(range(0, row_count))
        train_indices = indices[:train_row_count]
        valid_indices = indices[train_row_count:]

        val = {'x': train['x'][valid_indices, :, :], 't': train['t'][valid_indices, :],
               'yf': train['yf'][valid_indices, :], 'ycf': train['ycf'][valid_indices, :]}
        tr = {'x': train['x'][train_indices, :, :], 't': train['t'][train_indices, :],
               'yf': train['yf'][train_indices, :], 'ycf': train['ycf'][train_indices, :]}

        # Averaged errors over all experiments
        avg_train_f_error = 0
        avg_train_cf_error = 0
        avg_val_f_error = 0
        avg_val_cf_error = 0

        # Create lists for losses
        predictions_from_all_experiments = []
        for i in range(experiments):
            print('Experiment: ' + str(i) + ', training in progress! ')
            l, p = self.trainer(my_nn, my_session, train_step, tr, val, test, iterations, batch_size, i, dropout,
                                my_alpha, my_lambda, train_row_count, valid_row_count)
            losses_at_convergence = l[-1:][0]
            avg_train_f_error += losses_at_convergence[1]
            avg_train_cf_error += losses_at_convergence[3]
            avg_val_f_error += losses_at_convergence[5]
            avg_val_cf_error += losses_at_convergence[7]
            predictions_from_all_experiments.append(p)
        self.avg_train_f_error = avg_train_f_error/experiments
        self.avg_train_cf_error = avg_train_cf_error/experiments
        self.avg_val_f_error = avg_val_f_error/experiments
        self.acg_val_cf_error = avg_val_cf_error/experiments
        self.ate = self.get_ate(test, predictions_from_all_experiments)

    def trainer(self, network, session, training_step, train_data, validation_data, test_data, iterations, batch_size,
                exp_id, dropout, my_alpha, my_lambda, train_row_count, valid_row_count):
        treat_prob = np.mean(train_data['t'])

        # Setup dictionary to be fed
        dict_factual = {network.x: train_data['x'][:, :, exp_id], network.t: train_data['t'][:, exp_id].reshape(train_row_count, 1),
                        network.y_true: train_data['yf'][:, exp_id].reshape(train_row_count, 1), network.net_alpha: my_alpha,
                        network.net_lambda: my_lambda, network.net_treatment_prob: treat_prob,
                        network.net_dropout_layer_set_1: dropout[0], network.net_dropout_layer_set_2: dropout[1]}
        dict_counter_factual = {network.x: train_data['x'][:, :, exp_id], network.t: 1 - train_data['t'][:, exp_id].reshape(train_row_count, 1),
                                network.y_true: train_data['ycf'][:, exp_id].reshape(train_row_count, 1),
                                network.net_dropout_layer_set_1: dropout[0], network.net_dropout_layer_set_2: dropout[1]}
        dict_valid_factual = {network.x: validation_data['x'][:, :, exp_id], network.t: validation_data['t'][:, exp_id].reshape(valid_row_count, 1),
                              network.y_true: validation_data['yf'][:, exp_id].reshape(valid_row_count, 1), network.net_alpha: my_alpha,
                              network.net_lambda: my_lambda, network.net_treatment_prob: treat_prob,
                              network.net_dropout_layer_set_1: dropout[0], network.net_dropout_layer_set_2: dropout[1]}
        dict_valid_counter_factual = {network.x: validation_data['x'][:, :, exp_id], network.t: 1 - validation_data['t'][:, exp_id].reshape(valid_row_count, 1),
                                network.y_true: validation_data['ycf'][:, exp_id].reshape(valid_row_count, 1), network.net_dropout_layer_set_1: dropout[0],
                                network.net_dropout_layer_set_2: dropout[1]}

        # Initialize tensorflow variables
        session.run(tf.global_variables_initializer())

        # Losses
        losses=[]
        objective_function, factual_error, imbalance_error = session.run([network.tot_loss, network.pred_loss,
                                                                          network.imb_dist], feed_dict=dict_factual)
        counter_factual_error = session.run(network.pred_loss, feed_dict=dict_counter_factual)

        valid_objective_function, valid_factual_error, valid_imbalance_error = session.run([network.tot_loss, network.pred_loss,
                                                                          network.imb_dist], feed_dict=dict_valid_factual)
        valid_counter_factual_error = session.run(network.pred_loss, feed_dict=dict_valid_counter_factual)

        losses.append([objective_function, factual_error, imbalance_error, counter_factual_error,
                       valid_objective_function, valid_factual_error, valid_imbalance_error, valid_counter_factual_error])

        # Predictions
        preds = []

        # Convergence configuration
        gap = 0.05
        convergence_iterations = 1500
        train_f_list = []
        train_cf_list = []
        val_f_list = []
        val_cf_list = []

        # Train for multiple iterations
        for i in range(iterations):
            batch_indices = random.sample(range(0, train_row_count), batch_size)
            x_batch = train_data['x'][:, :, exp_id][batch_indices, :]
            t_batch = train_data['t'][:, exp_id][batch_indices]
            y_batch = train_data['yf'][:, exp_id][batch_indices]

            session.run(training_step, feed_dict={network.x: x_batch, network.t: t_batch.reshape(batch_size, 1), network.y_true: y_batch.reshape(batch_size, 1),
                        network.net_alpha: my_alpha, network.net_lambda: my_lambda, network.net_treatment_prob: treat_prob,
                        network.net_dropout_layer_set_1: dropout[0], network.net_dropout_layer_set_2: dropout[1]})

            # Loss
            obj, f_error, imb_error = session.run([network.tot_loss, network.pred_loss, network.imb_dist],
                                                  feed_dict=dict_factual)
            cf_error = session.run(network.pred_loss, feed_dict=dict_counter_factual)
            valid_obj, valid_f_error, valid_imb = session.run([network.tot_loss, network.pred_loss,
                                                               network.imb_dist], feed_dict=dict_valid_factual)
            valid_cf_error = session.run(network.pred_loss, feed_dict=dict_valid_counter_factual)

            losses.append([obj, f_error, imb_error, cf_error, valid_obj, valid_f_error, valid_imb, valid_cf_error])
            loss_str = str(i) + ' Obj: ' + str(obj) + ' Fact: ' + str(f_error) + ' CFact: ' \
                       + str(cf_error) + ' Imb: ' + str(imb_error) + ' VObj: ' + str(valid_obj) \
                       + ' VFact: ' + str(valid_f_error) + ' VCFact: ' + str(valid_cf_error) + ' VImb: ' + \
                       str(valid_imb)

            # Populating lists for checking convergence
            train_f_list.append(f_error)
            train_cf_list.append(cf_error)
            val_f_list.append(valid_f_error)
            val_cf_list.append(valid_cf_error)

            # Prediction
            test_size = test_data['x'].shape[0]
            y_pred_f_test = session.run(network.output, feed_dict={network.x: test_data['x'][:, :, exp_id],
                                                                   network.t: test_data['t'][:, exp_id].reshape(
                                                                       test_size, 1),
                                                                   network.net_dropout_layer_set_1: dropout[0],
                                                                   network.net_dropout_layer_set_2: dropout[1]})
            y_pred_cf_test = session.run(network.output, feed_dict={network.x: test_data['x'][:, :, exp_id],
                                                                    network.t: 1 - test_data['t'][:, exp_id].reshape(
                                                                        test_size, 1),
                                                                    network.net_dropout_layer_set_1: dropout[0],
                                                                    network.net_dropout_layer_set_2: dropout[1]})
            predictions = np.concatenate((y_pred_f_test, y_pred_cf_test), axis=1)
            preds.append(predictions)

            convergence_tracker = self.check_convergence([train_f_list, train_cf_list, val_f_list, val_cf_list],
                                                         convergence_iterations, gap)
            # Check convergence over 1000 iterations
            if convergence_tracker:
                loss_str += ' Converged'
                print(loss_str)
                break
            if i == iterations - 1:
                loss_str += ' Iteration overshoot'
                print(loss_str)
                break
            if i % 250 == 0:
                print(loss_str)
        return losses, preds

    def check_convergence(self, losses, convergence_iterations, gap):
        if len(losses[0]) >= convergence_iterations:
            f = losses[0][(-1*convergence_iterations):]
            cf = losses[1][(-1*convergence_iterations):]
            vf = losses[2][(-1*convergence_iterations):]
            vcf = losses[3][(-1*convergence_iterations):]
            variations_f = max(f) - min(f)
            variations_cf = max(cf) - min(cf)
            variations_vf = max(vf) - min(vf)
            variations_vcf = max(vcf) - min(vcf)
            if variations_f <= gap and variations_cf <= gap and variations_vf <= gap and variations_vcf <= gap:
                return True
            else:
                return False
        else:
            return False

    def get_ate(self, test, predictions):
        factualGroundTruth = test['yf']
        treatment = test['t']
        ate_treatment = 0
        ate_control = 0
        for i in range(len(predictions)):
            t = treatment[:, i]
            zeroIndices = np.where(t == 0)[0].tolist()
            nonZeroIndices = np.where(t == 1)[0].tolist()

            f = factualGroundTruth[:, i]
            pcf = predictions[i][-1:][0][:, 1]

            # If t = 0(factual)
            a = np.take(pcf, zeroIndices)
            b = np.take(f, zeroIndices)
            diff = a - b
            ate_control += np.mean(diff)

            # If t = 1(factual)
            a = np.take(f, nonZeroIndices)
            b = np.take(pcf, nonZeroIndices)
            diff = a - b
            ate_treatment += np.mean(diff)
        return (ate_control/len(predictions)+ate_treatment/len(predictions))/2
