import numpy as np

from env import CarWorld
from agent import train_q_learning
from visualize import print_summary, plot_training, animate_learning

def main():
    np.random.seed(42)
    
    env = CarWorld()
    
    q_table, rewards, steps_history, success_history, snapshot_paths, snapshot_meta = train_q_learning(
        env,
        episodes=600,
        alpha=0.18,
        gamma=0.96,
        epsilon=1.0,
        epsilon_decay=0.992,
        epsilon_min=0.05,
    )
    
    print_summary(rewards, steps_history, success_history)
    plot_training(rewards, steps_history, success_history)
    animate_learning(env, snapshot_paths, snapshot_meta)
    
if __name__ == "__main__":
    main()