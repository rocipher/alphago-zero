import numpy as np
import logging
import tempfile
import os
import tensorflow.contrib.keras.api.keras.layers as layers
import tensorflow.contrib.keras.api.keras.models as models
import tensorflow.contrib.keras.api.keras.losses as losses
import tensorflow.contrib.keras.api.keras.optimizers as optimizers
import tensorflow.contrib.keras.api.keras.metrics as metrics

from hyper_params import *
from go_board import *

class Model():
    def __init__(self):
        pass

    def train_on_hist_batch(self, hist_batch):
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

        self.model_id = model_id               
        input_layer = layers.Input(shape=(STATE_HIST_SIZE*2+1, BOARD_SIZE, BOARD_SIZE))
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
                        loss=[losses.mean_squared_error, losses.binary_crossentropy],
                        metrics=[metrics.binary_accuracy])
      

    def get_input_from_state(self, state: GameState):
        return np.concatenate((state.pos[::-1, :, :, :].reshape(STATE_HIST_SIZE*2, BOARD_SIZE, BOARD_SIZE),
                              np.repeat(state.player, BOARD_SIZE*BOARD_SIZE).reshape(1, BOARD_SIZE, BOARD_SIZE)), axis=0)

    def train_on_hist_batch(self, hist_batch):
        x = np.array([self.get_input_from_state(game_move.state) for game_move in hist_batch])
        y = [np.array([game_move.value for game_move in hist_batch]),
             np.array([game_move.action_distribution for game_move in hist_batch])]
        self.model.train_on_batch(x, y)

    def predict(self, state):
        # expand the batch dimension
        x = np.expand_dims(self.get_input_from_state(state), axis=0)
        value, action_probabilities = self.model.predict(x)
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
        conv_layer = layers.Conv2D(CONV_FILTERS, CONV_KERNEL, data_format="channels_first", padding='same')(input_layer)
        conv_layer = layers.BatchNormalization()(conv_layer)
        conv_layer = layers.Activation("relu")(conv_layer)
        return conv_layer

    def build_residual_block(self, input_layer):            
        reslayer = layers.Conv2D(CONV_FILTERS, CONV_KERNEL, data_format="channels_first", padding='same')(input_layer)
        reslayer = layers.BatchNormalization()(reslayer)
        reslayer = layers.Activation("relu")(reslayer)
        reslayer = layers.Conv2D(CONV_FILTERS, CONV_KERNEL, data_format="channels_first", padding='same')(reslayer)
        reslayer = layers.BatchNormalization()(reslayer)
        assert reslayer.shape.as_list() == input_layer.shape.as_list()
        reslayer = layers.Add()([reslayer, input_layer])
        reslayer = layers.Activation("relu")(reslayer)
        return reslayer
    
    def build_value_head(self, input_layer):
        value_head = layers.Conv2D(1, (1, 1), data_format="channels_first")(input_layer)        
        value_head = layers.BatchNormalization()(value_head)
        value_head = layers.Activation("relu")(value_head)
        value_head = layers.Flatten()(value_head)
        value_head = layers.Dense(DENSE_SIZE, "relu")(value_head)
        value_head = layers.Dense(1, "tanh")(value_head)
        return value_head

    def build_policy_head(self, input_layer):
        policy_head = layers.Conv2D(2, (1, 1), data_format="channels_first")(input_layer)
        policy_head = layers.BatchNormalization()(policy_head)
        policy_head = layers.Activation("relu")(policy_head)
        policy_head = layers.Flatten()(policy_head)
        policy_head = layers.Dense(BOARD_SIZE*BOARD_SIZE+1)(policy_head)
        policy_head = layers.Activation("softmax")(policy_head)
        return policy_head
