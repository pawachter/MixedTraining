import tensorflow as tf
from keras import layers
import keras


class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.projection = layers.Dense(units=projection_dim, kernel_initializer='he_normal')
        self.position_embedding = self.add_weight(
            shape=(1, num_patches, projection_dim),
            initializer='random_normal',
            trainable=True,
            name='position_embedding'
        )

    def call(self, patch):
        encoded_patch = self.projection(patch) + self.position_embedding
        encoded_patch = self.layer_norm(encoded_patch)
        return encoded_patch


class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        channels = tf.shape(images)[3]

        patch_dims = channels * self.patch_size * self.patch_size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patches = tf.reshape(patches, [tf.shape(images)[0], -1, patch_dims])
        return patches


def mlp(x, hidden_units, dense_layers):
    for units, layer in zip(hidden_units, dense_layers):
        x = layer(x)
    return x


class ViT(keras.Model):
    def __init__(self, config, img_size, n_classes):
        super(ViT, self).__init__()

        self.patch_size = config.patch_size
        self.projection_dim = config.projection_dim
        self.num_heads = config.num_heads
        self.transformer_layers = config.transformer_layers
        self.mlp_head_units = config.mlp_head_units
        self.transformer_units = [
            config.projection_dim * 2,
            config.projection_dim,
        ]
        self.num_patches = (img_size[0] // config.patch_size) ** 2
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.patches = Patches(self.patch_size)
        self.patch_encoder = PatchEncoder(self.num_patches, self.projection_dim)
        self.transformer_blocks = [self.transformer_block() for _ in range(self.transformer_layers)]
        self.flatten = layers.Flatten()
        self.mlp_head = [layers.Dense(units, activation=tf.nn.gelu, kernel_initializer='he_normal') for units in
                         self.mlp_head_units]
        self.classifier = layers.Dense(n_classes, activation=tf.nn.softmax, kernel_initializer='he_normal')

    def transformer_block(self):
        inputs = keras.Input(shape=(self.num_patches, self.projection_dim))
        inputs_normed = self.layer_norm(inputs)

        attention_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim)(
            inputs_normed, inputs_normed)
        x2 = layers.Add()([attention_output, inputs])
        x2 = self.layer_norm(x2)
        x3 = mlp(x2, hidden_units=self.transformer_units,
                 dense_layers=[layers.Dense(units, activation=tf.nn.gelu, kernel_initializer='he_normal') for units in
                               self.transformer_units])
        x = layers.Add()([x3, x2])
        return keras.Model(inputs=inputs, outputs=x)

    def call(self, inputs):
        patches = self.patches(inputs)
        encoded_patches = self.patch_encoder(patches)

        x = encoded_patches
        for block in self.transformer_blocks:
            x = block(x)
        representation = self.flatten(x)

        for layer in self.mlp_head:
            representation = layer(representation)
        outputs = self.classifier(representation)

        return outputs
