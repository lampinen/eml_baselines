import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1

import os
import re
#import cProfile
from copy import deepcopy
from itertools import permutations
from collections import Counter

from mini_imagenet_dataloader import MiniImageNetDataLoader
from orthogonal_matrices import random_orthogonal

### Parameters #################################################

config = {
    "run_offset": 0,
    "num_runs": 1,

    "vision_checkpoint": None, #"./resnet_v1_50.ckpt",
    "fine_tune_vision": True,
    "input_size": [224, 224, 3],

    "way": 5, # how many classes 
    "shot": 5, # how many shots
    "test_sample_num": 5, # how many test examples per class -- restricted by
                          # gpu memory for how many can efficiently run
                          

    "num_hidden": 128,
    "num_hidden_hyper": 1024,

    "M_max_pool": True,  # whether to max or average across examples in M
    "optimizer": "Adam",

    "init_learning_rate": 3e-6,

    "lr_decay": 0.8,

    "lr_decays_every": 250,
    "min_learning_rate": 1e-8,

    "max_epochs": 500000,
    "batches_per_epoch": 100,
    "num_task_hidden_layers": 3,
    "num_hyper_hidden_layers": 3,
    "train_drop_prob": 0.00, # dropout probability, applied on meta and hyper
                             # but NOT task or input/output at present. Note
                             # that because of multiplicative effects and depth
                             # impact can be dramatic.
    "train_vision_drop_prob": 0.5, # for vision output
    "start_vision_drop_epoch": 400, 
    "task_weight_weight_mult": 0.8, # not a typo, the init range of the final
                                   # hyper weights that generate the task
                                   # parameters. 


    # if a restore checkpoint path is provided, will restore from it instead of
    # running the initial training phase
    "restore_checkpoint_path": None, 
    "output_dir": "/mnt/fs4/lampinen/eml_baselines/mini_imagenet_224_redux/results6_%ishot_%iway/",
    "eval_every": 200, 
    "eval_batches": 50,
    "big_eval_every": 2000, 
    "big_eval_batches": 200,
    "save_every": 2000, # how often to save a checkpoint
    
    "train_base": True, 

    "internal_nonlinearity": tf.nn.leaky_relu,
    "output_nonlinearity": None
}
config["output_dir"] = config["output_dir"] % (config["shot"], config["way"])

### END PARAMATERS (finally) ##################################

def _save_config(filename, config):
    with open(filename, "w") as fout:
        fout.write("key, value\n")
        for key, value in config.items():
            fout.write(key + ", " + str(value) + "\n")

var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')

def get_distinct_random_choices(values, num_choices_per, num_sets,
                                replace=False):
    sets = []
    while len(sets) < num_sets:
        candidate_set = set(np.random.choice(values, num_choices_per, 
                                             replace=replace))
        if candidate_set not in sets:
            sets.append(candidate_set)
        
    return [np.random.permutation(list(s)) for s in sets]


class meta_model(object):
    """A meta-learning model for polynomials."""
    def __init__(self, config, dataloader):
        """args:
            config: a config dict, see above
            dataloader: a mini_imagenet data loader
        """
        self.config = config
        self.dataloader = dataloader

        # bookkeeping
        way = self.config["way"]
        shot = self.config["shot"]
        test_sample_num = self.config["test_sample_num"]
        self.dataset_length = way * (shot + test_sample_num)
        self.dataset_train_portion = way * shot
        self.dataset_test_portion = way * test_sample_num 
        self.num_output = way 
        self.tkp = 1. - config["train_drop_prob"] # drop prob -> keep prob
        self.tvkp = 1. - config["train_vision_drop_prob"] # drop prob -> keep prob
        self.train_idx = 0 
        self.eval_idx = {"train": 0, "val": 0, "test": 0}

        self.idx_limit = {
            "train": len(
                self.dataloader.train_filenames) // self.dataset_length,
            "val": len(self.dataloader.val_filenames) // self.dataset_length,
            "test": len(self.dataloader.test_filenames) // self.dataset_length}
        
        # network

        # vision input
        input_size = config["input_size"]
        output_size = config["way"]

        self.base_input_ph = tf.placeholder(
            tf.float32, shape=[None] + input_size)
        self.base_target_ph = tf.placeholder(
            tf.float32, shape=[None, output_size])

        self.lr_ph = tf.placeholder(tf.float32)
        self.keep_prob_ph = tf.placeholder(tf.float32) # dropout keep prob
        self.vision_keep_prob_ph = tf.placeholder(tf.float32) # for vision output 

        num_hidden = config["num_hidden"]
        num_hidden_hyper = config["num_hidden_hyper"]
        internal_nonlinearity = config["internal_nonlinearity"]
        output_nonlinearity = config["output_nonlinearity"]


        def _input_preprocessing(inputs):
            return tf.image.resize_bilinear(inputs,
                                            [224, 224])

        if input_size[0] != 224 or input_size[1] != 224:
            preprocessed_inputs = _input_preprocessing(self.base_input_ph)
        else:
            preprocessed_inputs = self.base_input_ph

        def _vision(preprocessed_inputs, reuse=True):
            with tf.variable_scope("vision", reuse=reuse):
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    resnet_output, _ = resnet_v1.resnet_v1_50(preprocessed_inputs,
                                                              is_training=True)
                if not config["fine_tune_vision"]:
                    resnet_output = tf.stop_gradient(resnet_output)
                resnet_output = tf.squeeze(resnet_output, axis=[1, 2])
                resnet_output = tf.nn.dropout(resnet_output,
                                              keep_prob=self.vision_keep_prob_ph)
                vision_result = slim.fully_connected(resnet_output, num_hidden_hyper,
                                                     activation_fn=None)
            return vision_result, resnet_output

                
        processed_input, resnet_output = _vision(preprocessed_inputs, 
                                                       reuse=False)
        if config["vision_checkpoint"]:
            tf.contrib.framework.init_from_checkpoint(
                config['vision_checkpoint'],
                {'resnet_v1_50/': 'vision/resnet_v1_50/'})

        self.processed_input = processed_input

        self.target_processor_nontf = random_orthogonal(num_hidden_hyper)[:, :output_size]
        self.target_processor = tf.get_variable('target_processor',
                                                shape=[num_hidden_hyper, output_size],
                                                initializer=tf.constant_initializer(self.target_processor_nontf))

        processed_targets = tf.matmul(self.base_target_ph, tf.transpose(self.target_processor))

        def _output_mapping(X):
            """hidden space mapped back to T/F output logits"""
            res = tf.matmul(X, self.target_processor)
            return res

        # function embedding "guessing" network / meta network
        # {(emb_in, emb_out), ...} -> emb
        self.guess_input_mask_ph = tf.placeholder(tf.bool, shape=[None]) # which datapoints get excluded from the guess

        def _meta_network(embedded_inputs, embedded_targets,
                          mask_ph=self.guess_input_mask_ph, reuse=True,
                          subscope=''):
            this_scope = 'meta/' + subscope + 'network'
            with tf.variable_scope(this_scope, reuse=reuse):
                guess_input = tf.concat([embedded_inputs,
                                         embedded_targets], axis=-1)
                guess_input = tf.boolean_mask(guess_input,
                                              self.guess_input_mask_ph)
                guess_input = tf.nn.dropout(guess_input, self.keep_prob_ph)

                gh_1 = slim.fully_connected(guess_input, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                gh_1 = tf.nn.dropout(gh_1, self.keep_prob_ph)
                gh_2 = slim.fully_connected(gh_1, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                gh_2 = tf.nn.dropout(gh_2, self.keep_prob_ph)
                if config["M_max_pool"]:
                    gh_2b = tf.reduce_max(gh_2, axis=0, keep_dims=True)
                else:
                    gh_2b = tf.reduce_mean(gh_2, axis=0, keep_dims=True)
                gh_3 = slim.fully_connected(gh_2b, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                gh_3 = tf.nn.dropout(gh_3, self.keep_prob_ph)

                guess_embedding = slim.fully_connected(gh_3, num_hidden_hyper,
                                                       activation_fn=None)
                guess_embedding = tf.nn.dropout(guess_embedding, self.keep_prob_ph)
                return guess_embedding

        self.guess_base_function_emb = _meta_network(processed_input,
                                                     processed_targets,
                                                     reuse=False)

        # hyper_network: emb -> (f: emb -> emb)
        self.feed_embedding_ph = tf.placeholder(np.float32,
                                                [1, num_hidden_hyper])

        num_task_hidden_layers = config["num_task_hidden_layers"]

        tw_range = config["task_weight_weight_mult"]/np.sqrt(
            num_hidden * num_hidden_hyper) # yields a very very roughly O(1) map
        task_weight_gen_init = tf.random_uniform_initializer(-tw_range,
                                                             tw_range)

        def _hyper_network(function_embedding, reuse=True, subscope=''):
            with tf.variable_scope('hyper' + subscope, reuse=reuse):
                hyper_hidden = function_embedding
                for _ in range(config["num_hyper_hidden_layers"]):
                    hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                        activation_fn=internal_nonlinearity)
                    hyper_hidden = tf.nn.dropout(hyper_hidden, self.keep_prob_ph)

                hidden_weights = []
                hidden_biases = []

                task_weights = slim.fully_connected(hyper_hidden, num_hidden*(num_hidden_hyper +(num_task_hidden_layers-1)*num_hidden + num_hidden_hyper),
                                                    activation_fn=None,
                                                    weights_initializer=task_weight_gen_init)
                task_weights = tf.nn.dropout(task_weights, self.keep_prob_ph)

                task_weights = tf.reshape(task_weights, [-1, num_hidden, (num_hidden_hyper + (num_task_hidden_layers-1)*num_hidden + num_hidden_hyper)])
                task_biases = slim.fully_connected(hyper_hidden, num_task_hidden_layers * num_hidden + num_hidden_hyper,
                                                   activation_fn=None)

                Wi = tf.transpose(task_weights[:, :, :num_hidden_hyper], perm=[0, 2, 1])
                bi = task_biases[:, :num_hidden]
                hidden_weights.append(Wi)
                hidden_biases.append(bi)
                for i in range(1, num_task_hidden_layers):
                    Wi = tf.transpose(task_weights[:, :, num_hidden_hyper+(i-1)*num_hidden:num_hidden_hyper+i*num_hidden], perm=[0, 2, 1])
                    bi = task_biases[:, num_hidden*i:num_hidden*(i+1)]
                    hidden_weights.append(Wi)
                    hidden_biases.append(bi)
                Wfinal = task_weights[:, :, -num_hidden_hyper:]
                bfinal = task_biases[:, -num_hidden_hyper:]

                for i in range(num_task_hidden_layers):
                    hidden_weights[i] = tf.squeeze(hidden_weights[i], axis=0)
                    hidden_biases[i] = tf.squeeze(hidden_biases[i], axis=0)

                Wfinal = tf.squeeze(Wfinal, axis=0)
                bfinal = tf.squeeze(bfinal, axis=0)
                hidden_weights.append(Wfinal)
                hidden_biases.append(bfinal)
                return hidden_weights, hidden_biases

        self.base_task_params = _hyper_network(self.guess_base_function_emb,
                                               reuse=False)
        self.fed_emb_task_params = _hyper_network(self.feed_embedding_ph)

        # task network
        def _task_network(task_params, processed_input):
            hweights, hbiases = task_params
            task_hidden = processed_input
            for i in range(num_task_hidden_layers):
                task_hidden = internal_nonlinearity(
                    tf.matmul(task_hidden, hweights[i]) + hbiases[i])

            raw_output = tf.matmul(task_hidden, hweights[-1]) + hbiases[-1]

            return raw_output

        self.base_raw_output = _task_network(self.base_task_params,
                                             processed_input)
        self.base_output = _output_mapping(self.base_raw_output)
        self.base_output_argmax = tf.argmax(self.base_output, axis=-1)

        self.base_raw_output_fed_emb = _task_network(self.fed_emb_task_params,
                                                     processed_input)
        self.base_output_fed_emb = _output_mapping(self.base_raw_output_fed_emb)
        self.base_output_fed_emb_argmax = tf.argmax(self.base_output_fed_emb, 
                                                    axis=-1)

        # loss
        loss_fn = tf.nn.softmax_cross_entropy_with_logits
        target_argmax = tf.argmax(self.base_target_ph, axis=-1)

        def _compute_accuracy(pred_ints, targ_ints):
            res = tf.equal(pred_ints, targ_ints)
            res = tf.reduce_mean(tf.cast(res, tf.float32))
            return res
            
        self.base_loss = loss_fn(logits=self.base_output, labels=self.base_target_ph)
        self.total_base_loss = tf.reduce_mean(self.base_loss)
        self.base_accuracy = _compute_accuracy(self.base_output_argmax, target_argmax)

        self.base_fed_emb_loss = loss_fn(logits=self.base_output_fed_emb, 
                                         labels=self.base_target_ph)
        self.total_base_fed_emb_loss = tf.reduce_mean(self.base_fed_emb_loss)
        self.base_fed_emb_accuracy = _compute_accuracy(
            self.base_output_fed_emb_argmax, target_argmax)

        if config["optimizer"] == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr_ph)
        elif config["optimizer"] == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr_ph)
        else:
            raise ValueError("Unknown optimizer: %s" % config["optimizer"])

        self.base_train = optimizer.minimize(self.total_base_loss)
        self.base_fed_emb_train = optimizer.minimize(self.total_base_fed_emb_loss)

        # Saver
        self.saver = tf.train.Saver()

        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())


    def _default_guess_mask(self):
        dataset_length = self.dataset_length 
        dataset_train_portion = self.dataset_train_portion
        mask = np.zeros(dataset_length, dtype=np.bool)
        mask[:dataset_train_portion] = True
        return mask


    def base_train_step(self, batch, lr, with_vision_dropout=True):
        train_img, train_label, test_img, test_label = batch
        vision_inputs = np.concatenate([train_img, test_img], axis=0)
        targets = np.concatenate([train_label, test_label], axis=0)
        feed_dict = {
            self.base_input_ph: vision_inputs,
            self.guess_input_mask_ph: self._default_guess_mask(),
            self.base_target_ph: targets,
            self.keep_prob_ph: self.tkp,
            self.vision_keep_prob_ph: self.tvkp if with_vision_dropout else 1.,
            self.lr_ph: lr
        }
        self.sess.run(self.base_train, feed_dict=feed_dict)


    def base_eval(self, batch):
        train_img, train_label, test_img, test_label = batch
        vision_inputs = np.concatenate([train_img, test_img], axis=0)
        targets = np.concatenate([train_label, test_label], axis=0)
        feed_dict = {
            self.base_input_ph: vision_inputs,
            self.guess_input_mask_ph: self._default_guess_mask(),
            self.base_target_ph: targets,
            self.keep_prob_ph: 1.,
            self.vision_keep_prob_ph: 1.,
        }
        fetches = [self.total_base_loss, self.base_accuracy]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        return res


    def run_base_eval(self, num_batches=500, include_test=True):
        phases = ["train", "val"]
        results = {}
        if include_test:
            phases.append("test")
        else:
            results["test"] = {"loss": -1,
                               "accuracy": -1}
        for phase in phases:
            loss = 0.
            accuracy = 0.
            for _ in range(num_batches):
                batch = self.dataloader.get_batch(phase=phase, 
                                                  idx=self.eval_idx[phase])
                this_loss, this_acc = self.base_eval(batch)
                loss += this_loss
                accuracy += this_acc
                self.eval_idx[phase] += 1
                if self.eval_idx[phase] >= self.idx_limit[phase]: 
                    self.eval_idx[phase] = 0
            loss /= num_batches
            accuracy /= num_batches
            results[phase] = {"loss": loss,
                              "accuracy": accuracy}

        return results 


    def get_base_embedding(self, batch):
        train_img, train_label, test_img, test_label = batch
        vision_inputs = np.concatenate([train_img, test_img], axis=0)
        targets = np.concatenate([train_label, test_label], axis=0)
        feed_dict = {
            self.base_input_ph: vision_inputs,
            self.guess_input_mask_ph: self._default_guess_mask(),
            self.base_target_ph: targets,
            self.keep_prob_ph: 1.,
            self.vision_keep_prob_ph: 1.,
        }
        res = self.sess.run(self.guess_base_function_emb, feed_dict=feed_dict)
        return res


    def save_parameters(self, filename):
        self.saver.save(self.sess, filename)


    def restore_parameters(self, filename):
        self.saver.restore(self.sess, filename)


    def run_training(self, filename_prefix):
        """Train model"""
        config = self.config
        loss_filename = filename_prefix + "_losses.csv"
        accuracy_filename = filename_prefix + "_accuracies.csv"

        def write_results(epoch, results, f_loss, f_acc, print_too=True):
            loss_results = f"{epoch}, {results['train']['loss']}, {results['val']['loss']}, {results['test']['loss']}\n"
            acc_results = f"{epoch}, {results['train']['accuracy']}, {results['val']['accuracy']}, {results['test']['accuracy']}\n"
            f_loss.write(loss_results)
            f_acc.write(acc_results)
            if print_too:
                print(loss_results)
                print(acc_results)

        with open(loss_filename, "w") as f_loss, open(accuracy_filename, "w") as f_acc:

            num_epochs = config["max_epochs"]
            batches_per_epoch = config["batches_per_epoch"]
            eval_every = config["eval_every"]
            big_eval_every = config["big_eval_every"]
            eval_batches = config["eval_batches"]
            big_eval_batches = config["big_eval_batches"]
            save_every = config["save_every"]

            learning_rate = config["init_learning_rate"]
            lr_decays_every = config["lr_decays_every"]
            lr_decay = config["lr_decay"]
            min_learning_rate = config["min_learning_rate"]

            f_loss.write("epoch, train_loss, val_loss, test_loss\n")
            f_acc.write("epoch, train_accuracy, val_accuracy, test_accuracy\n")
            results = self.run_base_eval(num_batches=big_eval_batches)
            write_results(0, results, f_loss, f_acc)

            for epoch in range(1, num_epochs+1):
                for batch_i in range(batches_per_epoch):
                    batch = self.dataloader.get_batch(phase="train", 
                                                      idx=self.train_idx)
                    self.base_train_step(batch, learning_rate, 
                                         with_vision_dropout=epoch > config["start_vision_drop_epoch"])
                    self.train_idx += 1
                    if self.train_idx >= self.idx_limit["train"]: 
                        self.train_idx = 0

                if epoch % eval_every == 0:
                    if epoch % big_eval_every == 0: 
                        num_batches = big_eval_batches
                    else:
                        num_batches = eval_batches
                    results = self.run_base_eval(num_batches=num_batches)
                    write_results(epoch, results, f_loss, f_acc)

                if epoch % save_every == 0:
                    self.save_parameters(filename_prefix + "_checkpoint")

                if epoch % lr_decays_every == 0 and epoch > 0:
                    if learning_rate > min_learning_rate:
                        learning_rate *= lr_decay



## data loading

dataloader = MiniImageNetDataLoader(shot_num=config["shot"], 
                                    way_num=config["way"],
                                    episode_test_sample_num=config["test_sample_num"])

dataloader.generate_data_list(phase='train')
dataloader.generate_data_list(phase='val')
dataloader.generate_data_list(phase='test')

dataloader.load_list(phase='all')

## running stuff

for run_i in range(config["run_offset"], config["run_offset"]+config["num_runs"]):

    np.random.seed(run_i)
    tf.set_random_seed(run_i)

    model = meta_model(config, dataloader=dataloader)

    filename_prefix = config["output_dir"] + "run%i" % run_i
    print("Now running %s" % filename_prefix)
    _save_config(filename_prefix + "_config.csv", config)

    if config["restore_checkpoint_path"] is not None:
        model.restore_parameters(config["restore_checkpoint_path"] + "run%i" % run_i + "_guess_checkpoint")
    else:
        model.run_training(filename_prefix=filename_prefix)

    model.save_parameters(filename_prefix + "_checkpoint")

#    model.save_embeddings(filename=filename_prefix + "_final_embeddings.csv",
#                          include_new=True)

    tf.reset_default_graph()
