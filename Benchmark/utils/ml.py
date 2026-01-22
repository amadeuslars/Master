import numpy as np
import random

class QLearningAgent:
    def __init__(self, num_destroy, num_repair, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha      
        self.gamma = gamma      
        self.epsilon = epsilon  
        
        # States: 0=Early, 1=Mid, 2=Late
        self.q_destroy = np.zeros((3, num_destroy))
        self.q_repair = np.zeros((3, num_repair))
        
        # Initialize with optimistic values to encourage exploration
        self.q_destroy.fill(5.0)
        self.q_repair.fill(5.0)
        
    def get_state(self, current_iter, max_iter):
        progress = current_iter / max_iter
        if progress < 0.33: return 0  
        if progress < 0.66: return 1  
        return 2                      

    def select_action(self, state, q_table):
        if random.random() < self.epsilon:
            return random.randint(0, len(q_table[state]) - 1) 
        else:
            values = q_table[state]
            random_noise = np.random.random(values.shape) * 1e-4 # Tiny noise for tie-breaking
            return np.argmax(values + random_noise)

    def update(self, state, action, reward, next_state, q_table):
        best_next = np.max(q_table[next_state])
        current_q = q_table[state][action]
        q_table[state][action] = current_q + \
                                 self.alpha * (reward + self.gamma * best_next - current_q)