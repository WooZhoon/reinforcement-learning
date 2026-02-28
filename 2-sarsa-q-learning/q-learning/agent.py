import random
from collections import defaultdict
from environment import Env


def arg_max(q_list):
    max_idx_list = []
    max_value = q_list[0]
    for idx, value in enumerate(q_list):
        if value > max_value:
            max_idx_list.clear()
            max_value = value
            max_idx_list.append(idx)
        elif value == max_value:
            max_idx_list.append(idx)
    return random.choice(max_idx_list)


class QLearningAgent:
    def __init__(self):
        self.actions = [0, 1, 2, 3]
        self.step_size = 0.01
        self.discount_factor = 0.99
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def learn(self, state, action, reward, next_state):
        state = str(state)
        next_state = str(next_state)
        q_1 = self.q_table[state][action]
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.step_size * (q_2 - q_1)

    def get_action(self, state):
        state = str(state)
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            action = arg_max(q_values)
        return action

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= 0.01


if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent()

    for episode in range(1000):
        state = env.reset()

        while True:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

            if done:
                agent.update_epsilon()
                env.print_value_all(agent.q_table)
                break
