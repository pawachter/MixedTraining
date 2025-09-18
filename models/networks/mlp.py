import keras
from keras import layers


class MLP(keras.models.Model):
    def __init__(self, config, img_size, n_classes):
        super(MLP, self).__init__()
        self.config = config
        self.n_classes = n_classes
        self.input_shape = img_size
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()

        for layer_config in self.config.layers:
            if layer_config.type == 'flatten':
                model.add(layers.Flatten(input_shape=self.input_shape))

            elif layer_config.type == 'dense' and layer_config.activation == "softmax":
                model.add(layers.Dense(
                    units=self.n_classes,
                    activation=layer_config.activation,
                    kernel_initializer="zeros"))

            elif layer_config.type == 'dense':
                model.add(layers.Dense(
                    units=layer_config.units,
                    activation=layer_config.activation,
                    kernel_initializer='he_uniform'))

        return model

    def call(self, inputs):
        return self.model(inputs)
