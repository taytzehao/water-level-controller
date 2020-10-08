import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

class model:
    def __init__(self, tank_diameter, tank_height, outflow_coefficient, max_pump_increase,
                 max_flow, time_per_episode=500,seed=42, time_stability_size=10):
        self.rand_generator = np.random.RandomState(seed)
        
        self.tank_diameter = tank_diameter
        self.tank_height = tank_height
        self.current_water_volume = (((self.tank_diameter / 2) ** 2) * np.pi) * self.tank_height * \
                                    self.rand_generator.uniform(low=0.1, high=0.9)

        self.outflow_coefficient = outflow_coefficient
        self.setpoint = self.rand_generator.uniform(low=0.2, high=0.8) * (((self.tank_diameter / 2) ** 2) * np.pi) * self.tank_height
        
        self.maximum_flow = max_flow
        self.current_flow_rate = 0
        self.time_step = 0

        self.state = (self.current_water_volume, self.current_flow_rate, self.setpoint,self.error())
        self.max_pump_increase = max_pump_increase

        self.time_per_episode = time_per_episode

    ## Provides current water height from the most up to date water volume
    def water_height(self):
        return self.current_water_volume / (((self.tank_diameter / 2) ** 2) * np.pi)

    ## Calculates the non-linear outflow rate of water from the tank
    def outflow(self):
        return np.sqrt(abs(self.water_height())+0.00000000001) * self.outflow_coefficient

    ## Error of water volume from current set point
    def error(self):
        error=(self.current_water_volume - self.setpoint)
        return error

    def step(self, pump_power):

        ## Flow rate of water can only be increased at a maximum rate of self.max_pump. This increases the difficulty by having a lag effect.
        self.current_flow_rate =self.current_flow_rate+ pump_power[0]

        ## Limit flow rate to a maximum of self.maximum_flow value and minimum of 0 flow rate
        self.current_flow_rate=np.clip(self.current_flow_rate,0,self.maximum_flow)
        
        ## Current water volume in the tank
        self.current_water_volume = self.current_water_volume + self.current_flow_rate - self.outflow()

        ## Determine whether it is a terminal state
        terminal = self.is_terminal(self.current_water_volume)

        ## Water volume can only be at 0 or maximum tank volume limit
        self.current_water_volume=np.clip(self.current_water_volume,0,((((self.tank_diameter / 2) ** 2) * np.pi) * self.tank_height))

        ## Reward is the square of volumetric error (m^3)^2
        reward = -((abs(self.current_water_volume - self.setpoint)/(((self.tank_diameter / 2) ** 2) * np.pi))**2 )

        ## Error of water volume from current set point
        self.time_step += 1

        return self.retrieve_observation(), reward, terminal

    ## the terminal state is reached if time step reaches more than 500 or if water level is at 2 extremes
    def is_terminal(self, water_vol):
        if self.time_step >= self.time_per_episode-1:
            return True
        else:
            return False

    ## Retrieve current state
    def retrieve_observation(self):

        self.state = (
            self.current_water_volume , self.current_flow_rate,
            self.setpoint ,
            self.error() 
        )
        return self.state

    ## Reset the current state of the water tank. This involves time_step, water volume, input flow rate of water and error
    def reset(self):

        ## Set point and current water volume are set to a random value
        self.setpoint = self.rand_generator.uniform(low=0.2, high=0.8) * (((self.tank_diameter / 2) ** 2) * np.pi) * self.tank_height
        self.current_water_volume = (((self.tank_diameter / 2) ** 2) * np.pi) * self.tank_height * self.rand_generator.uniform(low=0.1, high=0.9)
        
        self.time_step = 0
        self.current_flow_rate=0
        
        return self.retrieve_observation()
