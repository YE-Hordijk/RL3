import REINFORCE as R
import ActorCritic as AC
import gymnasium as gym
import argparse

#import Environment as Env
#observation = Env.env.reset()


#*******************************************************************************

class Learning():
    def __init__(self, args, max_episodes=300, LearningRate=0.001, gamma=0.99, epsilon=0.9, render_mode="human"):
        self.max_episodes = max_episodes #100 #300
        self.Env = gym.make("LunarLander-v2", render_mode="human")


#*******************************************************************************

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="REINFORCE", help="Algorithm Options: REINFORCE, ActorCritic")
    parser.add_argument("--bootstrapping", default="True", help="Bootstrapping Options: True, False")
    parser.add_argument("--baseline_subtraction", default="True", help="Baseline Subtraction Options: True, False")

    args = vars(parser.parse_args())
    print(args)
    print("poep")
    Learn = Learning(args)
