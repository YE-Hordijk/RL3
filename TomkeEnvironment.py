import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    print("\nobs:", observation)
    print("action:", action)
    print("reward:", reward)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
