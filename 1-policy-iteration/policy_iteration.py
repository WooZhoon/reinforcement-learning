from environment import Env, GraphicDisplay, HEIGHT, WIDTH

# Goal state [row, col]  (matches GOAL = [col=2, row=2] -> [row=2, col=2])
GOAL_STATE = [2, 2]


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.discount_factor = 0.9

        # Value table: HEIGHT x WIDTH, initialised to 0
        self.value_table = [[0.0] * WIDTH for _ in range(HEIGHT)]

        # Policy table: HEIGHT x WIDTH, each cell holds a probability
        # distribution over 4 actions [up, down, left, right].
        # Initially uniform: each action has probability 0.25.
        self.policy_table = self.init_policy()

    # ------------------------------------------------------------------ #
    #  Initialise policy
    # ------------------------------------------------------------------ #

    def init_policy(self):
        """Return a HEIGHT x WIDTH table where every cell is [0.25]*4."""
        table = []
        for row in range(HEIGHT):
            row_data = []
            for col in range(WIDTH):
                row_data.append([0.25, 0.25, 0.25, 0.25])
            table.append(row_data)
        return table

    # ------------------------------------------------------------------ #
    #  Policy Evaluation  (one full sweep)
    # ------------------------------------------------------------------ #

    def policy_evaluation(self):
        """Perform one sweep of iterative policy evaluation."""
        next_value_table = [[0.0] * WIDTH for _ in range(HEIGHT)]

        for state in self.env.get_all_states():
            row, col = state

            # Terminal (goal) state: value stays 0
            if state == GOAL_STATE:
                next_value_table[row][col] = 0.0
                continue

            value = 0.0
            for action in self.env.possible_actions:
                policy_prob = self.policy_table[row][col][action]
                next_state  = self.env.state_after_action(state, action)
                reward      = self.env.get_reward(state, action)
                next_val    = self.value_table[next_state[0]][next_state[1]]

                value += policy_prob * (reward + self.discount_factor * next_val)

            next_value_table[row][col] = round(value, 2)

        self.value_table = next_value_table

    # ------------------------------------------------------------------ #
    #  Policy Improvement  (greedy w.r.t. current value table)
    # ------------------------------------------------------------------ #

    def policy_improvement(self):
        """Update the policy greedily with respect to the current value table."""
        next_policy_table = self.init_policy()

        for state in self.env.get_all_states():
            row, col = state

            # Terminal state: leave as uniform (won't be used)
            if state == GOAL_STATE:
                next_policy_table[row][col] = [0.0, 0.0, 0.0, 0.0]
                continue

            # Compute Q-values for each action
            q_values = []
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward     = self.env.get_reward(state, action)
                next_val   = self.value_table[next_state[0]][next_state[1]]
                q_values.append(reward + self.discount_factor * next_val)

            # Find the maximum Q-value
            max_q = max(q_values)

            # Build a new policy: uniform over all actions that achieve max_q
            max_count = q_values.count(max_q)
            new_policy = []
            for q in q_values:
                if q == max_q:
                    new_policy.append(round(1.0 / max_count, 2))
                else:
                    new_policy.append(0.0)

            next_policy_table[row][col] = new_policy

        self.policy_table = next_policy_table

    # ------------------------------------------------------------------ #
    #  Helpers expected by GraphicDisplay
    # ------------------------------------------------------------------ #

    def get_value(self, row, col):
        return round(self.value_table[row][col], 2)

    def get_policy(self, row, col):
        return self.policy_table[row][col]


# ====================================================================== #
#  Main
# ====================================================================== #

if __name__ == "__main__":
    env    = Env()
    agent  = PolicyIteration(env)
    # GraphicDisplay needs a reference to env for move_by_policy
    agent.env = env
    app = GraphicDisplay(agent)
    app.mainloop()
