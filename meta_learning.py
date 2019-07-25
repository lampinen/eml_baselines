from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

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

    "vision_checkpoint": "./resnet_v1_50.ckpt",
    "fine_tune_vision": True,
    "input_size": [84, 84, 3],

    "way": 5, # how many classes 
    "shot": 5, # how many shots

    "num_hidden": 64,
    "num_hidden_hyper": 512,
    "num_hidden_language": 512,

    "num_lstm_layers": 2, # for language processing
    "max_sentence_len": 20, # Any longer than this will not be trained
    "optimizer": "Adam",

    "init_learning_rate": 3e-5,
    "init_language_learning_rate": 3e-5,
    "init_meta_learning_rate": 1e-5,

    "new_init_learning_rate": 1e-7,
    "new_init_language_learning_rate": 1e-7,
    "new_init_meta_learning_rate": 1e-7,

    "lr_decay": 0.85,
    "language_lr_decay": 0.8,
    "meta_lr_decay": 0.85,

    "lr_decays_every": 100,
    "min_learning_rate": 3e-8,

    "max_base_epochs": 4000,
    "num_task_hidden_layers": 3,
    "num_hyper_hidden_layers": 3,
    "train_drop_prob": 0.00, # dropout probability, applied on meta and hyper
                             # but NOT task or input/output at present. Note
                             # that because of multiplicative effects and depth
                             # impact can be dramatic.

    "task_weight_weight_mult": 1., # not a typo, the init range of the final
                                   # hyper weights that generate the task
                                   # parameters. 


    # if a restore checkpoint path is provided, will restore from it instead of
    # running the initial training phase
    "restore_checkpoint_path": None, 
    "output_dir": "/mnt/fs4/lampinen/eml_baselines/mini_imagenet/results_%ishot_%iway/",
    "save_every": 20, 

    "memory_buffer_size": 1024, # How many points for each polynomial are stored
    "early_stopping_thresh": 0.05,
    
    "train_base": True, 

    "internal_nonlinearity": tf.nn.leaky_relu,
    "output_nonlinearity": None
}

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
    def __init__(self, config):
        """args:
            config: a config dict, see above
        """
        self.config = config
        self.meta_batch_size = config["way"] * config["shot"]
        self.num_output = config["way"]
        self.tkp = 1. - config["train_drop_prob"] # drop prob -> keep prob
        
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

        num_hidden = config["num_hidden"]
        num_hidden_hyper = config["num_hidden_hyper"]
        internal_nonlinearity = config["internal_nonlinearity"]
        output_nonlinearity = config["output_nonlinearity"]


        def _input_preprocessing(inputs):
            return tf.image.resize_bilinear(inputs,
                                            [224, 224])

        preprocessed_inputs = _input_preprocessing(self.base_input_ph)

        def _vision(preprocessed_inputs, reuse=True):
            with tf.variable_scope("vision", reuse=reuse):
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    resnet_output, _ = resnet_v1.resnet_v1_50(preprocessed_inputs,
                                                              is_training=False)
                if not config["fine_tune_vision"]:
                    resnet_output = tf.stop_gradient(resnet_output)
                vision_result = slim.fully_connected(resnet_output, num_hidden_hyper,
                                                     activation_fn=None)
            return vision_result, resnet_output

                
        processed_input, resnet_output = _vision(preprocessed_inputs, 
                                                       reuse=False)
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
                gh_2b = tf.reduce_max(gh_2, axis=0, keep_dims=True)
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

        self.base_loss = loss_fn(logits=self.base_output, labels=self.base_target_ph)
        self.total_base_loss = tf.reduce_mean(self.base_loss)
        self.base_accuracy = tf.reduce_mean(tf.equal(self.base_output_argmax, target_argmax))

        self.base_fed_emb_loss = loss_fn(logits=self.base_output_fed_emb, 
                                         labels=self.base_target_ph)
        self.total_base_fed_emb_loss = tf.reduce_mean(self.base_fed_emb_loss)
        self.base_accuracy = tf.reduce_mean(
            tf.equal(self.base_output_fed_emb_argmax, target_argmax))

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
        exit()


    def _default_guess_mask(self):
        way = self.config["way"]
        shot = self.config["shot"]
        mask = np.zeros(dataset_length, dtype=np.bool)
        mask[:way * shot] = True
        return mask


    def base_train_step(self, memory_buffer, lr):
        input_buff, output_buff = memory_buffer.get_memories()
        feed_dict = {
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: self._random_guess_mask(self.memory_buffer_size),
            self.base_target_ph: output_buff,
            self.keep_prob_ph: self.tkp,
            self.lr_ph: lr
        }
        self.sess.run(self.base_train, feed_dict=feed_dict)


    def base_language_train_step(self, intified_task, memory_buffer, lr):
        input_buff, output_buff = memory_buffer.get_memories()
        feed_dict = {
            self.base_input_ph: input_buff,
            self.language_input_ph: intified_task,
            self.lang_keep_ph: self.lang_keep_prob,
            self.base_target_ph: output_buff,
            self.keep_prob_ph: self.tkp,
            self.lr_ph: lr
        }
        self.sess.run(self.base_lang_train, feed_dict=feed_dict)


    def base_eval(self, memory_buffer, meta_batch_size=None):
        input_buff, output_buff = memory_buffer.get_memories()
        feed_dict = {
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: self._random_guess_mask(
                self.memory_buffer_size, meta_batch_size=meta_batch_size),
            self.base_target_ph: output_buff,
            self.keep_prob_ph: 1.
        }
        fetches = [self.total_base_loss]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        return res


    def run_base_eval(self, include_new=False, sweep_meta_batch_sizes=False):
        """sweep_meta_batch_sizes: False or a list of meta batch sizes to try"""
        if include_new:
            tasks = self.all_base_tasks_with_implied
        else:
            tasks = self.initial_base_tasks_with_implied

        losses = [] 
        if sweep_meta_batch_sizes:
            for meta_batch_size in sweep_meta_batch_sizes:
                this_losses = [] 
                for task in tasks:
                    task_str = _stringify_polynomial(task)
                    memory_buffer = self.memory_buffers[task_str]
                    res = self.base_eval(memory_buffer, 
                                         meta_batch_size=meta_batch_size)
                    this_losses.append(res[0])
                losses.append(this_losses)
        else:
            for task in tasks:
                task_str = _stringify_polynomial(task)
                memory_buffer = self.memory_buffers[task_str]
                res = self.base_eval(memory_buffer)
                losses.append(res[0])

        names = [_stringify_polynomial(t) for t in tasks]
        return names, losses
        

    def base_language_eval(self, intified_task, memory_buffer):
        input_buff, output_buff = memory_buffer.get_memories()
        feed_dict = {
            self.base_input_ph: input_buff,
            self.language_input_ph: intified_task,
            self.base_target_ph: output_buff,
            self.lang_keep_ph: 1.,
            self.keep_prob_ph: 1.
        }
        fetches = [self.total_base_lang_loss]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        return res


    def run_base_language_eval(self, include_new=False):
        if include_new:
            tasks = self.all_base_tasks_with_implied
        else:
            tasks = self.initial_base_tasks_with_implied

        losses = [] 
        names = []
        for task in tasks:
            task_str = _stringify_polynomial(task)
            intified_task = self.task_to_ints[task_str]
#            print(task_str)
#            print(intified_task)
            if intified_task is None:
                continue
            memory_buffer = self.memory_buffers[task_str]
            res = self.base_language_eval(intified_task, memory_buffer)
            losses.append(res[0])
            names.append(task_str)

        return names, losses


    def base_embedding_eval(self, embedding, memory_buffer):
        input_buff, output_buff = memory_buffer.get_memories()
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.feed_embedding_ph: embedding,
            self.base_input_ph: input_buff,
            self.base_target_ph: output_buff
        }
        fetches = [self.total_base_fed_emb_loss]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        return res

    
    def get_base_embedding(self, memory_buffer):
        input_buff, output_buff = memory_buffer.get_memories()
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: np.ones([self.memory_buffer_size]),
            self.base_target_ph: output_buff
        }
        res = self.sess.run(self.guess_base_function_emb, feed_dict=feed_dict)
        return res


    def get_language_embedding(self, intified_task):
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.lang_keep_ph: 1.,
            self.language_input_ph: intified_task
        }
        res = self.sess.run(self.language_function_emb, feed_dict=feed_dict)
        return res


    def get_combined_embedding(self, t1_embedding, t2_embedding):
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.meta_input_ph: t1_embedding,
            self.meta_input_2_ph: t2_embedding
        }
        res = self.sess.run(self.combined_meta_inputs, feed_dict=feed_dict)
        return res


    def get_meta_dataset(self, meta_task, include_new=False):
        x_data = []
        x2_data = []
        y_data = []
        if include_new:
            this_base_tasks = self.meta_pairings_full[meta_task]
        else:
            this_base_tasks = self.meta_pairings_base[meta_task]
        for this_tuple in this_base_tasks:
            if len(this_tuple) == 3: # binary_func 
                task, task2, other = this_tuple
                task2_buffer = self.memory_buffers[task2]
                x2_data.append(self.get_base_embedding(task2_buffer)[0, :])
            else:
                task, other = this_tuple
            task_buffer = self.memory_buffers[task]
            x_data.append(self.get_base_embedding(task_buffer)[0, :])
            if other in [0, 1]:  # for classification meta tasks
                y_data.append([other])
            else:
                other_buffer = self.memory_buffers[other]
                y_data.append(self.get_base_embedding(other_buffer)[0, :])
        if x2_data != []: # binary func
            return {"x1": np.array(x_data), "x2": np.array(x2_data),
                    "y": np.array(y_data)}
        else:
            return {"x": np.array(x_data), "y": np.array(y_data)}


    def refresh_meta_dataset_cache(self, include_new=False):
        meta_tasks = self.all_base_meta_tasks 
        if include_new:
            meta_tasks += self.all_new_meta_tasks 

        for t in meta_tasks:
            self.meta_dataset_cache[t] = self.get_meta_dataset(t, include_new)


    def meta_loss_eval(self, meta_dataset):
        feed_dict = {
            self.keep_prob_ph: 1.,
        }
        y_data = meta_dataset["y"]
        if y_data.shape[-1] == 1:
            feed_dict[self.meta_class_ph] = y_data 
            fetch = self.total_meta_t_loss
        else:
            feed_dict[self.meta_target_ph] = y_data 
            fetch = self.total_meta_m_loss

        if "x1" in meta_dataset: 
            feed_dict[self.meta_input_ph] = meta_dataset["x1"]
            feed_dict[self.meta_input_2_ph] = meta_dataset["x2"]
            feed_dict[self.guess_input_mask_ph] =  np.ones([len(meta_dataset["x1"])])
            fetch = self.total_meta_bf_loss
        else:
            feed_dict[self.meta_input_ph] = meta_dataset["x"]
            feed_dict[self.guess_input_mask_ph] =  np.ones([len(meta_dataset["x"])])

        return self.sess.run(fetch, feed_dict=feed_dict)
        

    def run_meta_loss_eval(self, include_new=False):
        meta_tasks = self.all_base_meta_tasks 
        if include_new:
            meta_tasks = self.all_meta_tasks 

        names = []
        losses = []
        for t in meta_tasks:
            meta_dataset = self.meta_dataset_cache[t]
            if meta_dataset == {}: # new tasks aren't cached
                meta_dataset = self.get_meta_dataset(t, include_new) 
            loss = self.meta_loss_eval(meta_dataset)
            names.append(t)
            losses.append(loss)

        return names, losses


    def get_meta_embedding(self, meta_dataset):
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.guess_input_mask_ph: np.ones([len(meta_dataset["x"])])
        }
        y_data = meta_dataset["y"]
        if y_data.shape[-1] == 1:
            feed_dict[self.meta_class_ph] = y_data 
            fetch = self.guess_meta_t_function_emb
        else:
            feed_dict[self.meta_target_ph] = y_data 
            fetch = self.guess_meta_m_function_emb

        if "x1" in meta_dataset: 
            feed_dict[self.meta_input_ph] = meta_dataset["x1"]
            feed_dict[self.meta_input_2_ph] = meta_dataset["x2"]
            fetch = self.guess_meta_bf_function_emb
        else:
            feed_dict[self.meta_input_ph] = meta_dataset["x"]

        return self.sess.run(fetch, feed_dict=feed_dict)


    def get_meta_outputs(self, meta_dataset, new_dataset=None):
        """Get new dataset mapped according to meta_dataset, or just outputs
        for original dataset if new_dataset is None"""
        meta_class = meta_dataset["y"].shape[-1] == 1

        if new_dataset is not None:
            if "x" in meta_dataset:
                this_x = np.concatenate([meta_dataset["x"], new_dataset["x"]], axis=0)
                new_x = new_dataset["x"] 
                this_mask = np.zeros(len(this_x), dtype=np.bool)
                this_mask[:len(meta_dataset["x"])] = True # use only these to guess
            else:
                this_x1 = np.concatenate([meta_dataset["x1"], new_dataset["x1"]], axis=0)
                this_x2 = np.concatenate([meta_dataset["x2"], new_dataset["x2"]], axis=0)
                new_x = new_dataset["x1"] # for size only 
                this_mask = np.zeros(len(this_x1), dtype=np.bool)
                this_mask[:len(meta_dataset["x1"])] = True # use only these to guess

            if meta_class:
                this_y = np.concatenate([meta_dataset["y"], np.zeros([len(new_x)])], axis=0)
            else:
                this_y = np.concatenate([meta_dataset["y"], np.zeros_like(new_x)], axis=0)

        else:
            if "x" in meta_dataset:
                this_x = meta_dataset["x"]
                this_mask = np.ones(len(this_x), dtype=np.bool)
            else:
                this_x1 = meta_dataset["x1"]
                this_x2 = meta_dataset["x2"]
                this_mask = np.ones(len(this_x1), dtype=np.bool)
            this_y = meta_dataset["y"]

        feed_dict = {
            self.keep_prob_ph: 1.,
            self.guess_input_mask_ph: this_mask 
        }
        if meta_class:
            feed_dict[self.meta_class_ph] = this_y 
            this_fetch = self.meta_t_output 
        else:
            feed_dict[self.meta_target_ph] = this_y
            this_fetch = self.meta_m_output 

        if "x1" in meta_dataset: 
            feed_dict[self.meta_input_ph] = this_x1 
            feed_dict[self.meta_input_2_ph] = this_x2 
            fetch = self.meta_bf_output
        else:
            feed_dict[self.meta_input_ph] = this_x 

        res = self.sess.run(this_fetch, feed_dict=feed_dict)

        if new_dataset is not None:
            return res[len(meta_dataset["y"]):, :]
        else:
            return res


    def run_meta_true_eval(self, include_new=False):
        """Evaluates true meta loss, i.e. the accuracy of the model produced
           by the embedding output by the meta task"""
        if include_new:
            meta_tasks = self.base_meta_mappings + self.new_meta_mappings
            meta_pairings = self.meta_pairings_full
        else:
            meta_tasks = self.base_meta_mappings
            meta_pairings = self.meta_pairings_base
        meta_binary_funcs = self.base_meta_binary_funcs

        names = []
        losses = []
        for meta_task in meta_tasks:
            meta_dataset = self.meta_dataset_cache[meta_task]
            if meta_dataset == {}: # new tasks aren't cached
                meta_dataset = self.get_meta_dataset(meta_task, include_new) 

            for task, other in meta_pairings[meta_task]:
                task_buffer = self.memory_buffers[task]
                task_embedding = self.get_base_embedding(task_buffer)

                other_buffer = self.memory_buffers[other]

                mapped_embedding = self.get_meta_outputs(
                    meta_dataset, {"x": task_embedding})

                names.append(meta_task + ":" + task + "->" + other)
                this_loss = self.base_embedding_eval(mapped_embedding, other_buffer)[0]
                losses.append(this_loss)
        for meta_task in meta_binary_funcs:
            meta_dataset = self.meta_dataset_cache[meta_task]
            for task1, task2, other in meta_pairings[meta_task]:
                task1_buffer = self.memory_buffers[task1]
                task1_embedding = self.get_base_embedding(task1_buffer)
                task2_buffer = self.memory_buffers[task2]
                task2_embedding = self.get_base_embedding(task2_buffer)

                other_buffer = self.memory_buffers[other]

                mapped_embedding = self.get_meta_outputs(
                    meta_dataset, {"x1": task1_embedding,
                                   "x2": task2_embedding})

                names.append(meta_task + ":" + task1 + ":" + task2 + "->" + other)
                this_loss = self.base_embedding_eval(mapped_embedding, other_buffer)[0]
                losses.append(this_loss)

        return names, losses 


    def meta_train_step(self, meta_dataset, meta_lr):
        if "y" not in meta_dataset:
            print(meta_dataset)
        y_data = meta_dataset["y"]
        if "x" in meta_dataset:
            feed_dict = {
                self.keep_prob_ph: self.tkp,
                self.meta_input_ph: meta_dataset["x"],
                self.guess_input_mask_ph: np.ones([len(meta_dataset["x"])]),
                self.lr_ph: meta_lr
            }
            meta_class = y_data.shape[-1] == 1
            if meta_class:
                feed_dict[self.meta_class_ph] = y_data
                op = self.meta_t_train
            else:
                feed_dict[self.meta_target_ph] = y_data
                op = self.meta_m_train
        else:
            feed_dict = {
                self.keep_prob_ph: self.tkp,
                self.meta_input_ph: meta_dataset["x1"],
                self.meta_input_2_ph: meta_dataset["x2"],
                self.guess_input_mask_ph: np.ones([len(meta_dataset["x1"])]),
                self.lr_ph: meta_lr
            }
            feed_dict[self.meta_target_ph] = y_data
            op = self.meta_bf_train

        self.sess.run(op, feed_dict=feed_dict)


    def save_parameters(self, filename):
        self.saver.save(self.sess, filename)


    def restore_parameters(self, filename):
        self.saver.restore(self.sess, filename)


    def run_training(self, filename_prefix, num_epochs, include_new=False):
        """Train model on base and meta tasks, if include_new include also
        the new ones."""
        config = self.config
        loss_filename = filename_prefix + "_losses.csv"
        sweep_filename = filename_prefix + "_sweep_losses.csv"
        meta_filename = filename_prefix + "_meta_true_losses.csv"
        lang_filename = filename_prefix + "_language_losses.csv"
        train_language = config["train_language"]
        train_base = config["train_base"]
        train_meta = config["train_meta"]

        with open(loss_filename, "w") as fout, open(meta_filename, "w") as fout_meta, open(lang_filename, "w") as fout_lang:
            base_names, base_losses = self.run_base_eval(
                include_new=include_new)
            meta_names, meta_losses = self.run_meta_loss_eval(
                include_new=include_new)
            meta_true_names, meta_true_losses = self.run_meta_true_eval(
                include_new=include_new)

            fout.write("epoch, " + ", ".join(base_names + meta_names) + "\n")
            fout_meta.write("epoch, " + ", ".join(meta_true_names) + "\n")

            base_loss_format = ", ".join(["%f" for _ in base_names]) + "\n"
            loss_format = ", ".join(["%f" for _ in base_names + meta_names]) + "\n"
            meta_true_format = ", ".join(["%f" for _ in meta_true_names]) + "\n"

            if train_language:
                (base_lang_names, 
                 base_lang_losses) = self.run_base_language_eval(
                    include_new=include_new)
                lang_loss_format = ", ".join(["%f" for _ in base_lang_names]) + "\n"
                fout_lang.write("epoch, " + ", ".join(base_lang_names) + "\n")

            s_epoch  = "0, "
            curr_losses = s_epoch + (loss_format % tuple(
                base_losses + meta_losses))
            curr_meta_true = s_epoch + (meta_true_format % tuple(
                meta_true_losses))
            fout.write(curr_losses)
            fout_meta.write(curr_meta_true)

            if train_language:
                curr_lang_losses = s_epoch + (lang_loss_format % tuple(
                    base_lang_losses))
                fout_lang.write(curr_lang_losses)

            if config["sweep_meta_batch_sizes"] is not None:
                with open(sweep_filename, "w") as fout_sweep:
                    sweep_names, sweep_losses = self.run_base_eval(
                        include_new=include_new, sweep_meta_batch_sizes=config["sweep_meta_batch_sizes"])
                    fout_sweep.write("epoch, size, " + ", ".join(base_names) + "\n")
                    for i, swept_batch_size in enumerate(config["sweep_meta_batch_sizes"]):
                        swept_losses = s_epoch + ("%i, " % swept_batch_size) + (base_loss_format % tuple(sweep_losses[i]))
                        fout_sweep.write(swept_losses)

            if include_new:
                tasks = self.all_tasks_with_implied
                learning_rate = config["new_init_learning_rate"]
                language_learning_rate = config["new_init_language_learning_rate"]
                meta_learning_rate = config["new_init_meta_learning_rate"]
            else:
                tasks = self.all_initial_tasks_with_implied
                learning_rate = config["init_learning_rate"]
                language_learning_rate = config["init_language_learning_rate"]
                meta_learning_rate = config["init_meta_learning_rate"]

            save_every = config["save_every"]
            early_stopping_thresh = config["early_stopping_thresh"]
            lr_decays_every = config["lr_decays_every"]
            lr_decay = config["lr_decay"]
            language_lr_decay = config["language_lr_decay"]
            meta_lr_decay = config["meta_lr_decay"]
            min_learning_rate = config["min_learning_rate"]
            min_meta_learning_rate = config["min_meta_learning_rate"]
            min_language_learning_rate = config["min_language_learning_rate"]


            self.fill_buffers(num_data_points=config["memory_buffer_size"],
                              include_new=True)
            self.refresh_meta_dataset_cache(include_new=include_new)
            for epoch in range(1, num_epochs+1):
                if epoch % config["refresh_mem_buffs_every"] == 0:
                    self.fill_buffers(num_data_points=config["memory_buffer_size"],
                                      include_new=True)
                if epoch % config["refresh_meta_cache_every"] == 0:
                    self.refresh_meta_dataset_cache(include_new=include_new)

                order = np.random.permutation(len(tasks))
                for task_i in order:
                    task = tasks[task_i]
                    if isinstance(task, str):
                        if train_meta:
                            dataset = self.meta_dataset_cache[task]
                            self.meta_train_step(dataset, meta_learning_rate)
                    else:
                        str_task = _stringify_polynomial(task)
                        memory_buffer = self.memory_buffers[str_task]
                        if train_base:
                            self.base_train_step(memory_buffer, learning_rate)
                        if train_language:
                            intified_task = self.task_to_ints[str_task]
                            if intified_task is not None:
                                self.base_language_train_step(
                                    intified_task, memory_buffer,
                                    language_learning_rate)


                if epoch % save_every == 0:
                    s_epoch  = "%i, " % epoch
                    _, base_losses = self.run_base_eval(
                        include_new=include_new)
                    _, meta_losses = self.run_meta_loss_eval(
                        include_new=include_new)
                    _, meta_true_losses = self.run_meta_true_eval(
                        include_new=include_new)
                    curr_losses = s_epoch + (loss_format % tuple(
                        base_losses + meta_losses))
                    curr_meta_true = s_epoch + (meta_true_format % tuple(meta_true_losses))
                    fout.write(curr_losses)
                    fout_meta.write(curr_meta_true)
                    if train_language:
                        (_, base_lang_losses) = self.run_base_language_eval(
                            include_new=include_new)
                        curr_lang_losses = s_epoch + (lang_loss_format % tuple(
                            base_lang_losses))
                        fout_lang.write(curr_lang_losses)
                        print(curr_losses, curr_lang_losses)
                        if np.all(curr_losses < early_stopping_thresh) and np.all(curr_lang_losses < early_stopping_thresh):
                            print("Early stop!")
                            break
                    else:
                        print(curr_losses)
                        if np.all(curr_losses < early_stopping_thresh):
                            print("Early stop!")
                            break


                if epoch % lr_decays_every == 0 and epoch > 0:
                    if learning_rate > min_learning_rate:
                        learning_rate *= lr_decay

                    if meta_learning_rate > min_meta_learning_rate:
                        meta_learning_rate *= meta_lr_decay

                    if train_language and language_learning_rate > min_language_learning_rate:
                        language_learning_rate *= language_lr_decay


            if config["sweep_meta_batch_sizes"] is not None:
                with open(sweep_filename, "a") as fout_sweep:
                    sweep_names, sweep_losses = self.run_base_eval(
                        include_new=include_new, sweep_meta_batch_sizes=config["sweep_meta_batch_sizes"])
                    for i, swept_batch_size in enumerate(config["sweep_meta_batch_sizes"]):
                        swept_losses = s_epoch + ("%i, " % swept_batch_size) + (base_loss_format % tuple(sweep_losses[i]))
                        fout_sweep.write(swept_losses)



## data loading

#dataloader = MiniImageNetDataLoader(shot_num=config["shot"], 
#                                    way_num=config["way"],
#                                    episode_test_sample_num=15)
#
#dataloader.load_list(phase='all')

#batch = dataloader.get_batch(phase='train', idx=0)
#for i in range(len(batch)):
#    print(batch[i].shape)
#
#exit()

## running stuff

for run_i in range(config["run_offset"], config["run_offset"]+config["num_runs"]):
    model = meta_model(config)

    np.random.seed(run_i)
    tf.set_random_seed(run_i)

    model = meta_model(config)

    filename_prefix = config["output_dir"] + "run%i" % run_i
    print("Now running %s" % filename_prefix)
    _save_config(filename_prefix + "_config.csv", config)


#    model.save_embeddings(filename=filename_prefix + "_init_embeddings.csv",
#                          include_new=False)
    if config["restore_checkpoint_path"] is not None:
        model.restore_parameters(config["restore_checkpoint_path"] + "run%i" % run_i + "_guess_checkpoint")
    else:
        model.run_training(filename_prefix=filename_prefix,
                           num_epochs=config["max_base_epochs"],
                           include_new=False)
#    cProfile.run('model.run_training(filename_prefix=filename_prefix, num_epochs=config["max_base_epochs"], include_new=False)')

    model.save_parameters(filename_prefix + "_guess_checkpoint")
#    model.save_embeddings(filename=filename_prefix + "_guess_embeddings.csv",
#                          include_new=True)

    model.run_training(filename_prefix=filename_prefix + "_new",
                       num_epochs=config["max_new_epochs"],
                       include_new=True)
#    cProfile.run('model.run_training(filename_prefix=filename_prefix + "_new", num_epochs=config["max_new_epochs"], include_new=True)')
    model.save_parameters(filename_prefix + "_final_checkpoint")

#    model.save_embeddings(filename=filename_prefix + "_final_embeddings.csv",
#                          include_new=True)

    tf.reset_default_graph()
