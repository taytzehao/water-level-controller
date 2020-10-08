import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def actor(input_state_shape,output_bound):

    last_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

    inputs = layers.Input(shape=(input_state_shape))

    out = layers.Dense(256,kernel_regularizer=tf.keras.regularizers.l1(), kernel_initializer = tf.keras.initializers.GlorotNormal())(inputs)
    out = tf.keras.layers.Dropout(0.2)(out)
    out = layers.LeakyReLU(alpha=0.3)(out)
    out = layers.BatchNormalization()(out)

    out = layers.Dense(256,kernel_regularizer=tf.keras.regularizers.l1(1), kernel_initializer = tf.keras.initializers.GlorotNormal())(out)
    out = layers.BatchNormalization()(out)

    outputs = layers.Dense(1, activation="tanh", kernel_initializer =last_init)(out)
    outputs = outputs*output_bound

    model = tf.keras.Model(inputs, outputs)

    return model

def critic(input_state_shape, input_action_shape):

    # State as input
    state_input = layers.Input(shape=(input_state_shape))
    state_out = layers.Dense(16,kernel_regularizer=tf.keras.regularizers.l1())(state_input)
    state_out = layers.BatchNormalization()(state_out)


    state_out = layers.LeakyReLU(alpha=0.3)(state_out)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(32,kernel_regularizer=tf.keras.regularizers.l1())(state_out)

    state_out = layers.LeakyReLU(alpha=0.3)(state_out)
    state_out = layers.BatchNormalization()(state_out)

    # Action as input
    action_input = layers.Input(shape=(input_action_shape))
    action_out = layers.Dense(32,kernel_regularizer=tf.keras.regularizers.l1(2))(action_input)
    action_out = tf.keras.layers.Dropout(0.2)(action_out)
    action_out = layers.LeakyReLU(alpha=0.3)(action_out)
    action_out = layers.BatchNormalization()(action_out)

    # Processed action and state inputs are concatenated here
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu",kernel_regularizer=tf.keras.regularizers.l1(3))(concat)
    out = layers.LeakyReLU(alpha=0.3)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(256,kernel_regularizer=tf.keras.regularizers.l1(3),activation="tanh")(out)
    out = layers.LeakyReLU(alpha=0.3)(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None,decay_constant=0.9998):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        self.decay_constant=decay_constant
        self.decay_num=1

    def __call__(self):
        ## Ornstein_Uhlenbeck noise formula
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x

        ## Decay noise as the simulation proceeds and as the model stabilizes. This helps to reduce model to act more greedily during the later stages and obtain optimal state.
        self.decay_num=self.decay_num*self.decay_constant
      
        return x*max(0.1,self.decay_num)

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

def policy(model,state, noise_object,output_bound):
    
    ## Obtain action from base actor and clip action before adding in noise
    action = tf.squeeze(model(state))
    clipped_action = np.clip(action, -output_bound, output_bound)

    noise = noise_object()

    # Adding noise to action
    sampled_actions = clipped_action+ noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, -output_bound, output_bound)

    return [np.squeeze(legal_action)],noise




