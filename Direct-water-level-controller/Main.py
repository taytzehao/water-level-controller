from DDPG_Model import OUActionNoise,actor,critic,policy
from Environment import model
from Buffer import Buffer,update_target_single
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

##Volume of 392m3
Tank_model=model(tank_diameter=10,tank_height=5,
                 outflow_coefficient=1.4,max_pump_increase=1.25,
                 max_flow=5,seed=42)

def min_max(max,min,val):
    return (val-min)/(max-min)

## Initialize base and critic model
base_actor=actor(input_state_shape=len(Tank_model.retrieve_observation()),output_bound=Tank_model.max_pump_increase)
base_critic=critic(input_state_shape=len(Tank_model.retrieve_observation()),input_action_shape=1)

## Initialize target actor to be identical as base actor
target_actor=actor(input_state_shape=len(Tank_model.retrieve_observation()),output_bound=Tank_model.max_pump_increase)
target_actor.set_weights(base_actor.get_weights())

## Initialize target critic to be identical as base critic
target_critic=critic(input_state_shape=len(Tank_model.retrieve_observation()),input_action_shape=1)
target_critic.set_weights(base_critic.get_weights())

## Initialize actor and critic adam optimizer
actor_optimizer=tf.keras.optimizers.Adam(0.0005, amsgrad=True)
critic_optimizer=tf.keras.optimizers.Adam(0.001, amsgrad=True)

## Initialize weights for exploration with decay
Noise=OUActionNoise(mean=np.array([0]), std_deviation=float(3) * np.ones(1),theta=3)

## Initialize experience buffer replay
buffer=Buffer(num_states=len(Tank_model.retrieve_observation()),num_actions=1,batch_size=64,prioritized_replay=True)

ep_reward_list=[]
Noise_mem=[]


gamma=0.99
tau=0.5

for episode in range(50):
    error_list = []

    Last_state=Tank_model.reset()
    average_reward=0

    ## Reduce value of states to 0-1. This helps to increase learning speed asthe values are of similar range to output.
    Last_state = list(Last_state)
    Last_state[0] = min_max(max=392.7, min=0, val=Last_state[0])
    Last_state[1] = min_max(max=Tank_model.maximum_flow, min=0, val=Last_state[1])
    Last_state[2] = min_max(max=392.7, min=0, val=Last_state[2])
    Last_state[3] = min_max(max=392.7, min=0, val=Last_state[3])
    Last_state = tuple(Last_state)

    while True:

        tf_last_state= tf.expand_dims(tf.convert_to_tensor(Last_state), 0)

        ## Obtain action and noise value as a record. Inject noise into action and convert PID gains into action with error
        action,noise= policy(model=base_actor,state=tf_last_state,noise_object=Noise,output_bound=Tank_model.max_pump_increase)

        ## Take action in environment
        state,reward,Terminal=Tank_model.step(action)

        state=list(state)
        state[0] = min_max(max=392.7, min=0, val=state[0])
        state[1] = min_max(max=Tank_model.maximum_flow, min=0, val=state[1])
        state[2] = min_max(max=392.7, min=0, val=state[2])
        state[3] = min_max(max=392.7, min=0, val=state[3])
        state=tuple(state)

        error_list.append(- abs(Tank_model.current_water_volume - Tank_model.setpoint))

        ## Record information of state
        buffer.record((Last_state, action, reward, state),base_critic=base_critic,base_actor=base_actor,target_actor=target_actor,target_critic=target_critic,gamma=gamma,reward=reward)

        ## Update base actor and base critic
        base_actor,base_critic=buffer.learn(base_critic=base_critic,base_actor=base_actor,target_actor=target_actor,
                                            target_critic=target_critic,gamma=gamma,actor_optimizer=actor_optimizer,
                                            critic_optimizer=critic_optimizer)

        ## Soft update target actor and target critic
        target_actor,target_critic=update_target_single(tau,base_critic=base_critic,base_actor=base_actor,target_actor=target_actor,target_critic=target_critic)

        ## Remember noise produced
        Noise_mem.append(noise)

        # End this episode when terminal is true
        if Terminal:
            break

        ## Update last state 
        Last_state = state

    Episode_average_reward = sum(error_list)/len(error_list)
    ep_reward_list.append(Episode_average_reward)
    print("Episode * {} * Avg Reward is ==> {}".format(episode, ep_reward_list[episode]))

    if (episode%49)==0:
        fig, ax = plt.subplots(7, 1)
        ax[0].plot(buffer.TD_error_memory,label="TD error",color="b")
        ax[1].plot(buffer.value_function_memory,label="value function",color="g")
        ax[2].plot(buffer.state_buffer[:buffer.buffer_counter][:,0], label="current water volume",color="r")
        ax[2].plot(buffer.state_buffer[:buffer.buffer_counter][:,2], label="setpoint",color="c")
        ax[3].plot(buffer.state_buffer[:buffer.buffer_counter][:,1], label="water inflow rate",color="m")
        ax[4].plot(buffer.state_buffer[:buffer.buffer_counter][:,3], label="current error",color="y")
        ax[5].plot(Noise_mem, label="Noise memory",color="b")
        ax[6].plot(ep_reward_list, label="episodic_error", color="k")
        fig.legend()
        plt.savefig("Direct DDPG controller prioritized experience")
        with open('Direct DDPG controller prioritized experience pickle.obj', 'wb') as file:
            pickle.dump(fig, file)
        plt.show()
        plt.close(fig)

base_actor.save("Direct DDPG controller actor prioritized experience.h5")
base_critic.save("Direct DDPG controller critic prioritized experience.h5")
target_actor.save("Direct DDPG controller target actor prioritized experience.h5")
target_critic.save("Direct DDPG controller target critic prioritized experience.h5")



