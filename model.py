import numpy as np
import logging
import tempfile
import os
import datetime
import tensorflow.contrib.keras.api.keras.layers as layers
import tensorflow.contrib.keras.api.keras.models as models
import tensorflow.contrib.keras.api.keras.losses as losses
import tensorflow.contrib.keras.api.keras.optimizers as optimizers
import tensorflow.contrib.keras.api.keras.metrics as metrics
import tensorflow.contrib.keras.api.keras.callbacks as callbacks
import tensorflow.contrib.keras.api.keras.regularizers as regularizers
import tensorflow.contrib.keras.api.keras.backend as K
import tensorflow as tf

from hyper_params import *
from go_board import *

_TF_SESSION_SET = False

class Model():
    def __init__(self):
        pass

    def set_seed(self):
        pass

    def train_on_hist_batch(self, hist_batch, batch_index):
        pass

    def train(self):
        pass

    def predict(self):        
        pass
        
    def copy(self):
        pass

    def save(self, filePath):
        pass
    

class SimpleNNModel(Model):
    def __init__(self, model_id=None):
        super(SimpleNNModel, self).__init__()

        self.init_tensorflow_session()

        self.model_id = model_id               
        self.data_format = "channels_first"
        self.batch_norm_axis = 1
        self.reg_alpha = 1e-4

        input_layer = layers.Input(shape=(STATE_HIST_SIZE*2+1, BOARD_SIZE, BOARD_SIZE), name="input")
        # NCHW -> NHWC
        # top_layer = layers.Permute((2, 3, 1))(input_layer)
        top_layer = self.build_convolutional_block(input_layer)
        for _ in np.arange(NUM_RESIDUAL_BLOCKS):
            top_layer = self.build_residual_block(top_layer)

        logging.debug("init: top_layer shape: %s", top_layer.shape)

        value_head = self.build_value_head(top_layer)
        policy_head = self.build_policy_head(top_layer)

        logging.debug("init: value_head shape: %s", value_head.shape)
        logging.debug("init: policy_head shape: %s", policy_head.shape)
        
        self.model = models.Model(inputs=[input_layer], outputs=[value_head, policy_head])
        self.model.compile(optimizer="adam",
                        loss=[losses.mean_squared_error, losses.categorical_crossentropy],
                        metrics=[])
        model_log_id = "eb-%s-%dx%d-%d-%d" % (datetime.datetime.now().strftime("%m%d%H%M"), BOARD_SIZE, BOARD_SIZE, MCTS_STEPS, BATCH_SIZE)
        self.tb_callback = callbacks.TensorBoard(log_dir="%s/%s" % (OUTPUT_DIR, model_log_id))
        self.tb_callback.set_model(self.model)
          
    def init_tensorflow_session(self):
        global _TF_SESSION_SET
        if _TF_SESSION_SET:
            return 
        logging.info("Initializing the tensorflow Session...")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=TENSORFLOW_GPU_MEM_ALLOC_FRACT)
        config = tf.ConfigProto(
                            device_count={"GPU": TENSORFLOW_GPU_COUNT, "CPU": TENSORFLOW_CPU_COUNT}, 
                            log_device_placement=False,
                            gpu_options=gpu_options)
        session = tf.Session(config=config)
        K.set_session(session)
        _TF_SESSION_SET = True

    def set_seed(self, seed):
        tf.set_random_seed(seed)
        np.random.seed(seed)

    def write_log(self, callback, names, logs, batch_no):
        # https://gist.github.com/joelthchao/ef6caa586b647c3c032a4f84d52e3a11
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()

    def get_input_from_state(self, state: GameState):
        return np.concatenate((state.pos[::-1, :, :, :].reshape(STATE_HIST_SIZE*2, BOARD_SIZE, BOARD_SIZE),
                              np.repeat(state.player, BOARD_SIZE*BOARD_SIZE).reshape(1, BOARD_SIZE, BOARD_SIZE)), axis=0)

    def train_on_hist_batch(self, hist_batch, batch_index):
        x = np.array([self.get_input_from_state(game_move.state) for game_move in hist_batch])
        y = [np.array([game_move.value for game_move in hist_batch]),
             np.array([game_move.action_distribution for game_move in hist_batch])]
        logs = self.model.train_on_batch(x, y)
        self.write_log(self.tb_callback, self.model.metrics_names, logs, batch_index)
        

    def predict(self, state):            
        multiple = isinstance(state, list)
        if multiple:
            x = np.array([self.get_input_from_state(s) for s in state])
        else:
            # expand the batch dimension
            x = np.expand_dims(self.get_input_from_state(state), axis=0)

        value, action_probabilities = self.model.predict(x)

        if multiple:
            return value, action_probabilities
        else:
            return value[0], action_probabilities[0]

    def copy(self):        
        weights_file = tempfile.NamedTemporaryFile(delete=False)
        weights_file.close()
        new_model = None
        try:                        
            self.model.save_weights(weights_file.name)
            new_model = SimpleNNModel(model_id=self.model_id)
            new_model.model.load_weights(weights_file.name)
        finally:
            os.unlink(weights_file.name)
        return new_model

    def save(self, filePath):
        self.model.save_weights(filePath)

    def load(self, filePath):
        self.model.load_weights(filePath)

    def build_convolutional_block(self, input_layer):            
        conv_layer = layers.Conv2D(CONV_FILTERS, CONV_KERNEL, data_format=self.data_format, 
                    padding='same', kernel_regularizer=regularizers.l2(self.reg_alpha))(input_layer)
        conv_layer = layers.BatchNormalization(axis=self.batch_norm_axis)(conv_layer)
        conv_layer = layers.Activation("relu")(conv_layer)
        return conv_layer

    def build_residual_block(self, input_layer):            
        reslayer = layers.Conv2D(CONV_FILTERS, CONV_KERNEL, data_format=self.data_format, 
                    padding='same', kernel_regularizer=regularizers.l2(self.reg_alpha))(input_layer)
        reslayer = layers.BatchNormalization(axis=self.batch_norm_axis)(reslayer)
        reslayer = layers.Activation("relu")(reslayer)
        reslayer = layers.Conv2D(CONV_FILTERS, CONV_KERNEL, data_format=self.data_format, 
                    padding='same', kernel_regularizer=regularizers.l2(self.reg_alpha))(reslayer)
        reslayer = layers.BatchNormalization(axis=self.batch_norm_axis)(reslayer)
        assert reslayer.shape.as_list() == input_layer.shape.as_list()
        reslayer = layers.Add()([reslayer, input_layer])
        reslayer = layers.Activation("relu")(reslayer)
        return reslayer
    
    def build_value_head(self, input_layer):
        value_head = layers.Conv2D(1, (1, 1), data_format=self.data_format,
                            kernel_regularizer=regularizers.l2(self.reg_alpha))(input_layer)        
        value_head = layers.BatchNormalization(axis=self.batch_norm_axis)(value_head)
        value_head = layers.Activation("relu")(value_head)
        value_head = layers.Flatten()(value_head)
        value_head = layers.Dense(DENSE_SIZE, "relu", 
                            kernel_initializer='random_uniform',
                            bias_initializer='ones',
                            kernel_regularizer=regularizers.l2(self.reg_alpha))(value_head)
        value_head = layers.Dense(1, "tanh", name="output_value", 
                            kernel_initializer='random_uniform',
                            bias_initializer='ones', 
                            kernel_regularizer=regularizers.l2(self.reg_alpha))(value_head)
        return value_head

    def build_policy_head(self, input_layer):
        policy_head = layers.Conv2D(2, (1, 1), data_format=self.data_format, 
                            kernel_regularizer=regularizers.l2(self.reg_alpha))(input_layer)
        policy_head = layers.BatchNormalization(axis=self.batch_norm_axis)(policy_head)
        policy_head = layers.Activation("relu")(policy_head)
        policy_head = layers.Flatten()(policy_head)
        policy_head = layers.Dense(BOARD_SIZE*BOARD_SIZE+1, 
                            kernel_initializer='random_uniform',
                            bias_initializer='ones',
                            kernel_regularizer=regularizers.l2(self.reg_alpha))(policy_head)
        policy_head = layers.Activation("softmax", name="output_policy")(policy_head)
        return policy_head


if __name__ == '__main__':
    m = SimpleNNModel()
    m.model.summary()