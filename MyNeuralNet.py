import tensorflow as tf
import numpy as np
import MyUtility


class MyNeuralNet():
    def __init__(self, x, t, y_true, net_alpha, net_lambda, net_dropout_layer_set_1, net_dropout_layer_set_2,
                 net_treatment_prob, params):
        self.x = x
        self.t = t
        self.y_true = y_true
        self.net_alpha = net_alpha
        self.net_lambda = net_lambda
        self.net_dropout_layer_set_1 = net_dropout_layer_set_1
        self.net_dropout_layer_set_2 = net_dropout_layer_set_2
        self.net_treatment_prob = net_treatment_prob

        self.variables = {}
        self.wd_loss = 0
        self._build_network(x, t, y_true, net_treatment_prob, net_alpha, net_lambda, net_dropout_layer_set_1,
                            net_dropout_layer_set_2, params)

    # Adds a variable to the internal 'variables' dict with variable name as the key and variable expression as the
    # value
    def _add_variable(self, var, name):
        self.variables[name] = var

    # Create a variable
    def _create_variable(self, var, name):
        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    # Create a variable with an impact(l2 loss) over weight decay loss
    def _create_variable_with_weight_decay(self, initializer, name, wd):
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    # The build network method which takes the feature vector, treatment, true output, treatment probability
    # and parameters as the input
    def _build_network(self, x, t, y_true, p_t,net_alpha, net_lambda, net_dropout_layer_set_1, net_dropout_layer_set_2,
                       params):
        # Initialize weights and biases
        weights_in = []
        biases_in = []

        # Initializing the hypothesis with x
        h_in = [x]
        # Construct input layers
        for iter in range(params.num_layers[0]):
            # Configuring the first layer
            if iter == 0:
                weights_in.append(tf.Variable(tf.random_normal([params.input_dimensions, params.num_hidden_nodes[0]],
                                                               stddev=params.init_weights / np.sqrt(params.input_dimensions))))
            else:
                weights_in.append(tf.Variable(tf.random_normal([params.num_hidden_nodes[0], params.num_hidden_nodes[0]],
                                                               stddev=params.init_weights / np.sqrt(params.num_hidden_nodes[0]))))
            biases_in.append(tf.Variable(tf.zeros([1, params.num_hidden_nodes[0]])))
            z = tf.matmul(h_in[iter], weights_in[iter]) + biases_in[iter]
            h_in.append(tf.nn.relu(z))
            h_in[iter+1] = tf.nn.dropout(h_in[iter+1], net_dropout_layer_set_1)
        h_rep = h_in[len(h_in)-1]

        # Construct output layers
        h_out = [tf.concat([h_rep, t], 1)]
        weights_out = []
        biases_out = []
        for iter in range(params.num_layers[1]):
            if iter == 0:
                w = tf.random_normal([h_out[0].shape[1].value, params.num_hidden_nodes[1]],
                                     stddev=params.init_weights / np.sqrt(h_out[0].shape[1].value))
                weights_out.append(self._create_variable_with_weight_decay(w, 'w_out' + str(iter), 1.0))
            else:
                w = tf.random_normal([params.num_hidden_nodes[1], params.num_hidden_nodes[1]],
                                     stddev=params.init_weights / np.sqrt(params.num_hidden_nodes[1]))
                weights_out.append(self._create_variable_with_weight_decay(w, 'w_out' + str(iter), 1.0))
            biases_out.append(tf.Variable(tf.zeros([1, params.num_hidden_nodes[1]])))
            z = tf.matmul(h_out[iter], weights_out[iter]) + biases_out[iter]
            h_out.append(tf.nn.relu(z))
            h_out[iter+1] = tf.nn.dropout(h_out[iter+1], net_dropout_layer_set_2)

        # Get predictions
        weights_pred = self._create_variable(tf.Variable((tf.random_normal([params.num_hidden_nodes[1], 1],
                                                                           stddev=params.init_weights / np.sqrt(params.num_hidden_nodes[1])))), 'w_pred')
        bias_pred = self._create_variable(tf.Variable(tf.zeros([1])), 'b_pred')
        h_pred = h_out[-1]
        y_pred = tf.matmul(h_pred, weights_pred) + bias_pred

        # Compute risk
        w_t = t / (2 * p_t)
        w_c = (1 - t) / (2 * (1 - p_t))
        sample_weight = w_t + w_c
        risk = tf.reduce_mean(sample_weight * tf.square(y_true - y_pred))

        # Compute factual prediction error and IPM
        pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
        imb_error = MyUtility.Utility.get_ipm(h_rep, params.p_ipm, t) * net_alpha

        # Objective function to be minimized
        total_error = risk + imb_error + net_lambda * self.wd_loss

        # Setting output to variables
        self.output = y_pred  # Prediction
        self.weights_in = weights_in  # Weights of input network
        self.weights_out = weights_out  # Weights of output network
        self.weights_pred = weights_pred  # Prediction weights
        self.h_rep = h_rep  # Phi
        self.tot_loss = total_error  # Objective function to be minimised
        self.pred_loss = pred_error  # Prediction loss
        self.imb_dist = imb_error  # Imbalance between treatment and control distributions
