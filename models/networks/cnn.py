import keras
from keras import layers


class AlexNet(keras.models.Model):
    def __init__(self, config, img_size, n_classes):
        super(AlexNet, self).__init__()
        self.config = config
        self.input_shape = img_size
        self.n_classes = n_classes
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()

        for layer_config in self.config.layers:  # ['layers']:
            if layer_config.type == 'conv2d':
                model.add(layers.Conv2D(
                    filters=layer_config.filters,
                    kernel_size=layer_config.kernel_size,
                    strides=layer_config.strides,
                    padding=layer_config.padding,
                    activation=layer_config.activation,
                    input_shape=self.input_shape if not model.layers else None
                ))
            elif layer_config.type == 'maxpooling2d':
                model.add(layers.MaxPooling2D(
                    pool_size=layer_config.pool_size,
                    strides=layer_config.strides,
                    padding=layer_config.padding
                ))
            elif layer_config.type == 'flatten':
                model.add(layers.Flatten())
            elif layer_config.type == 'dense' and layer_config.activation == 'softmax':
                model.add(layers.Dense(
                    units=self.n_classes,
                    activation=layer_config.activation
                ))
            elif layer_config.type == 'dense':
                model.add(layers.Dense(
                    units=layer_config.units,
                    activation=layer_config.activation
                ))

        return model

    def call(self, inputs):
        return self.model(inputs)