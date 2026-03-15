import math
import numpy as np

class CarWorld:
    def __init__(self):
        self.width = 10.0
        self.height = 10.0
        
        self.start = np.array([1.0, 1.0], dtype=np.float32)
        self.goal = np.array([8.5, 8.5], dtype=np.float32)
        self.goal_radius = 0.7
        
        self.max_steps = 120
        self.speed = 0.45
        
        self.actions = np.array([
            [0.0, 1.0],    # up
            [0.7, 0.7],    # up-right
            [1.0, 0.0],    # right
            [0.7, -0.7],   # down-right
            [0.0, -1.0],   # down
            [-0.7, -0.7],  # down-left
            [-1.0, 0.0],   # left
            [-0.7, 0.7],   # up-left
        ], dtype=np.float32)
        
        self.obstacles = [
            (2.0, 2.5, 1.2, 4.0),
            (4.5, 0.5, 1.0, 5.0),
            (6.3, 4.5, 1.3, 4.0),
            (2.5, 7.0, 3.0, 1.0),
        ]
        
        self.grid_size = 20
        self.reset()
        
    def reset(self):
        self.pos = self.start.copy()
        self.heading = 0.0
        self.steps = 0
        return self.get_state()
    
    @property
    def n_actions(self):
        return len(self.actions)
    
    @property
    def n_states(self):
        return self.grid_size * self.grid_size
    
    def distance_to_goal(self, pos=None):
        p = self.pos if pos is None else pos
        return float(np.linalg.norm(p - self.goal()))
    
    def in_bounds(self, pos):
        return 0.0 <= pos[0] <= self.width and 0.0 <= pos[1] <= self.height
    
    def hits_obstacle(self, pos):
        x , y = float(pos[0]), float(pos[1])
        for ox, oy, ow, oh in self.obstacles:
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                return True
        return False
    
    def reached_goal(self, pos=None):
        p = self.pos if pos is None else pos
        return np.linalg.norm(p - self.goal()) <= self.goal_radius
    
    def get_state(self, action_idx):
        self.steps += 1
        
        action = self.actions[action_idx]
        action = action / np.linalg.norm(action)
        
        old_dist = self.distance_to_goal()
        candidate = self.pos + action * self.speed
        
        reward = -0.15
        done = False
        
        if not self.in_bounds(candidate):
            reward = -4.0
            candidate = self.pos.copy()
        elif self.hits_obstacle(candidate):
            reward = -6.0
            candidate = self.pos.copy()
        else:
            new_dist = self.distance_to_goal(candidate)
            reward += (old_dist - new_dist) * 1.8
            self.pos = candidate
            self.heading = math.atan2(action[1], action[0])

        if self.reached_goal():
            reward = 25.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self.get_state(), reward, done