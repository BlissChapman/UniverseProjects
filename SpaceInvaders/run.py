import gym
import universe  # register the universe environments

from agent import RandomAgent

# testing comment
env = gym.make("SpaceInvaders-v0")

env.seed(0)
agent = RandomAgent(env.action_space)

episode_count = 100
reward = 0
done = False

for i in range(episode_count):
    ob = env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)

        env.render()
        if done:
            print("Episode Complete")
            break
