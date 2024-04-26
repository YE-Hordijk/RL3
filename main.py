import REINFORCE as R
import ActorCritic as AC
import gymnasium as gym
import argparse

import matplotlib
import matplotlib.pyplot as plt
#import Environment as Env
#observation = Env.env.reset()


#***************************

class Learning():
    def _init_(self, args, render_mode="human"):
        self.max_episodes = 100 #300
        self.Env = gym.make("LunarLander-v2", render_mode="human")


#***************************
    '''
    def Reinforce_Learn(pi, theta, eta):
        initialize theta
        while not converged:
            grad = 0
            for m in range(M)
                sample trace h_0{s_0,a_0,r_0,s_1,...,s_n+1} according to policy pi(a|s)
                R = 0
                for t in reversed(range(n))
                    R = r_t + self.gamma * R
                    grad += R* rho log pi(a_t|s_t)
            theta <- theta + eta * grad
        return pi
    '''
#***************************
    '''
    def ActorCritic_Learn(pi, V_phi, n, eta, M): \\maybe met bootstrapping
        initialize theta, phi
        while not converged:
            grad = 0
            for m in range(M)
                sample trace h_0{s_0,a_0,r_0,s_1,...,s_n+1} according to policy pi(a|s)
                for t in range(T)
                    Q(s,a) = sum(r_t+k + V_phi(s_t+n)^2) van k=0 tot n-1
            phi = phi - eta * rho som(Q(s,a)-V_phi(s_t)^2)
            phi = phi - eta * rho som(Q(s,a)-V_phi(s_t)^2)
        return pi
    '''
#***************************

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="REINFORCE", help="Algorithm Options: REINFORCE, ActorCritic")
    parser.add_argument("--bootstrapping", default="True", help="Bootstrapping Options: True, False")
    parser.add_argument("--baseline_subtraction", default="True", help="Baseline Subtraction Options: True, False")

    args = vars(parser.parse_args())
    print(args)


    r = R.REINFORCE() # initialize the models
    policy = r.Reinforce_Learn()
    print("end", policy)
    
    plt.plot(range(len(policy)), policy)
    plt.xlabel("Episode")
    plt.ylabel("Timestep")
    plt.show()
    plt.savefig("test.png")
