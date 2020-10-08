import numpy as np
import tensorflow as tf

class Buffer:
    def __init__(self,num_states ,num_actions, buffer_capacity=100000, batch_size=64,priority_scale=1,prioritized_replay=False,importance_weight_bias=1,importance_weight_bias_geometric_constant=0.99):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))


        ## Memory for actor and critic loss memory
        
        self.TD_error_memory = np.array([])
        self.value_function_memory = np.array([])

        ## Initialize priority for  prioritized experience replay
        self.error_preferences=np.zeros((self.buffer_capacity, 1))
        self.importance_weight_bias=importance_weight_bias
        self.importance_weight_bias_geometric_constant=importance_weight_bias_geometric_constant
        self.prioritized_replay=prioritized_replay
        self.priority_scale = priority_scale

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple,base_actor=None,base_critic=None,target_actor=None,target_critic=None,gamma=None,reward=None):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]


        if self.prioritized_replay==True:
            state = tf.expand_dims(tf.convert_to_tensor(obs_tuple[0]),axis=0)
            action = tf.expand_dims(tf.keras.backend.expand_dims(tf.convert_to_tensor(obs_tuple[1])),axis=0)
            next_state=tf.expand_dims(tf.convert_to_tensor(obs_tuple[3]),axis=0)

            ## Calculate target value of the next state
            target_action = target_actor([next_state])
            target_value = reward + gamma * target_critic([next_state, target_action])
            
            ## Calculate value of current state
            base_value = base_critic([state, action])
            
            ## Calculate and store loss value in error preferences so that it can be used as a probability basis for choosing this state
            loss = target_value - base_value
            self.error_preferences[index]=abs(loss)+0.000000000001
        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self,base_actor,base_critic,target_actor,target_critic,critic_optimizer,actor_optimizer,gamma):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        
        # Randomly sample indices
        if self.prioritized_replay==False:
            batch_indices = np.random.choice(record_range, self.batch_size)
        else:
            ## Experience is sampled with error as priority. importance weight bias is used to normalize samples that have a high probability rate of being sampled. 
            scale_priority=self.error_preferences**self.priority_scale
            error_probabilities=np.squeeze(scale_priority[:record_range]/np.sum(scale_priority),axis=1)
            batch_indices = np.random.choice(record_range, self.batch_size,p=error_probabilities)

            ## Importance weight bias is used to normalize samples that have a high probability rate of being sampled. 
            ## This helps to reduce biasness towards samples with higher errors.
            importance_weight = ((1 / self.buffer_counter) * (1 / error_probabilities[batch_indices])) ** (1 - self.importance_weight_bias)

            ## Update importance weight bias
            self.importance_weight_bias = self.importance_weight_bias * self.importance_weight_bias_geometric_constant
        
        # Convert recorded values to tensors.
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])



        # First gradient tape to update critic network.
        with tf.GradientTape() as tape:

            ##Calculate target value of sampled batch
            actions=target_actor([next_state_batch])
            target_value = reward_batch + gamma * \
                           target_critic([next_state_batch,actions])
            base_value = base_critic([state_batch,action_batch])
            
            ## Loss value is multiplied to importance weight when prioritized replay is utilized.
            if self.prioritized_replay== False:
                loss = tf.math.reduce_mean(tf.math.square((target_value - base_value)))
            else:
                loss = tf.math.reduce_mean((tf.math.square((target_value- base_value)))*importance_weight)

        grad = tape.gradient(loss, base_critic.trainable_variables)

        critic_optimizer.apply_gradients(
            zip(grad, base_critic.trainable_variables)
        )
        '''
        if self.prioritized_replay == True:

            ## Update error preference with new error preferences.
            self.error_preferences[batch_indices] = tf.expand_dims(tf.squeeze(abs(target_value - base_value)),axis=1)
        '''
        ## Second gradient tape to update actor network
        with tf.GradientTape(persistent=True) as tape2:

            actions = base_actor([state_batch])
            critic_value = base_critic([state_batch, actions])
            actor_loss = -critic_value

            ## Obtain gradient of direct controller output from actor loss variable        


        ## Apply gradient of direct_controller_loss to base actor      
        actor_grad = tape2.gradient(actor_loss, base_actor.trainable_variables)

        actor_optimizer.apply_gradients(
            zip(actor_grad, base_actor.trainable_variables)
        )

        ## Update value for value function and TD error memory
        self.value_function_memory = np.append(self.value_function_memory,loss)
        self.TD_error_memory=np.append(self.TD_error_memory,actor_loss.numpy())
        
        return base_actor,base_critic

## Soft update tau
def update_target_single(tau,base_actor,base_critic,target_actor,target_critic):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(base_critic.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables =target_actor.weights
    for i, variable in enumerate(base_actor.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_actor.set_weights(new_weights)
    return target_actor,target_critic



