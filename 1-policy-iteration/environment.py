from tkinter import *
from tkinter import ttk
import time

# Grid parameters
UNIT = 60         # pixels per cell
HEIGHT = 10       # grid height (rows)
WIDTH = 10        # grid width (columns)

# Goal position
GOAL = [2, 2]

# 24 obstacle positions [col, row]
OBSTACLES = [
    [1, 2], [1, 3], [1, 6], [1, 8],
    [2, 1], [2, 4], [2, 8],
    [3, 1], [3, 6], [3, 7],
    [4, 2], [4, 4], [4, 8],
    [5, 4], [5, 6],
    [6, 1], [6, 2], [6, 5], [6, 8],
    [7, 1], [7, 8],
    [8, 3], [8, 4], [8, 6],
]


class GraphicDisplay(Tk):
    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('Policy Iteration (10x10)')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT + 110, HEIGHT * UNIT + 110))
        self.texts = []
        self.arrows = []
        self.agent = agent
        self.canvas = self._build_canvas()
        self.evaluation_count = 0
        self.improvement_count = 0

        # Buttons
        ttk.Button(self, text="Evaluate",
                   command=self.evaluate).grid(row=HEIGHT * UNIT // 60 + 1,
                                               column=0)
        ttk.Button(self, text="Improve",
                   command=self.improve).grid(row=HEIGHT * UNIT // 60 + 1,
                                              column=1)
        ttk.Button(self, text="move",
                   command=self.move).grid(row=HEIGHT * UNIT // 60 + 2,
                                           column=0)
        ttk.Button(self, text="reset",
                   command=self.reset).grid(row=HEIGHT * UNIT // 60 + 2,
                                            column=1)

    def _build_canvas(self):
        canvas = Canvas(self, bg='white',
                        height=HEIGHT * UNIT,
                        width=WIDTH * UNIT)

        # Draw grid lines
        for col in range(0, WIDTH * UNIT, UNIT):   # vertical lines
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, HEIGHT * UNIT, UNIT):  # horizontal lines
            x0, y0, x1, y1 = 0, row, WIDTH * UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        # Load images
        self.rectangle = PhotoImage(Image=None)
        self.triangle   = PhotoImage(Image=None)
        self.circle     = PhotoImage(Image=None)

        try:
            self.rectangle = PhotoImage(file="img/rectangle.png")
            self.triangle  = PhotoImage(file="img/triangle.png")
            self.circle    = PhotoImage(file="img/circle.png")
        except Exception:
            pass

        # Draw obstacles (triangles)
        for obs in OBSTACLES:
            col, row = obs
            canvas.create_image(col * UNIT + UNIT // 2,
                                 row * UNIT + UNIT // 2,
                                 image=self.triangle)
            canvas.create_text(col * UNIT + UNIT // 2,
                                row * UNIT + UNIT // 2 + 16,
                                text="R : -1.0", font=("Arial", 6))

        # Draw goal (circle) at GOAL = [2, 2]
        canvas.create_image(GOAL[0] * UNIT + UNIT // 2,
                             GOAL[1] * UNIT + UNIT // 2,
                             image=self.circle)
        canvas.create_text(GOAL[0] * UNIT + UNIT // 2,
                            GOAL[1] * UNIT + UNIT // 2 + 16,
                            text="R : 1.0", font=("Arial", 6))

        # Draw agent (rectangle) at start position [0, 0]
        self.rectangle_item = canvas.create_image(UNIT // 2, UNIT // 2,
                                                   image=self.rectangle)

        canvas.pack()
        return canvas

    # ------------------------------------------------------------------ #
    #  Value / reward text helpers
    # ------------------------------------------------------------------ #

    def text_value(self, row, col, contents, font='Arial', size=10,
                   style='normal', anchor='nw'):
        origin_x = UNIT // 2
        origin_y = UNIT // 2
        x = origin_x + (UNIT * col)
        y = origin_y + (UNIT * row)
        font_spec = (font, size, style)
        return self.canvas.create_text(x, y, fill='black', text=contents,
                                       font=font_spec, anchor=anchor)

    def text_reward(self, row, col, contents, font='Arial', size=10,
                    style='normal', anchor='nw'):
        origin_x = UNIT // 2
        origin_y = UNIT // 2
        x = origin_x + (UNIT * col)
        y = origin_y + (UNIT * row)
        font_spec = (font, size, style)
        return self.canvas.create_text(x, y, fill='black', text=contents,
                                       font=font_spec, anchor=anchor)

    # ------------------------------------------------------------------ #
    #  Arrow helpers
    # ------------------------------------------------------------------ #

    def draw_one_arrow(self, col, row, policy):
        if col == GOAL[0] and row == GOAL[1]:
            return

        action_map = {
            0: "img/up.png",
            1: "img/down.png",
            2: "img/left.png",
            3: "img/right.png",
        }

        # Find the action with maximum probability
        if max(policy) != 0:
            action = policy.index(max(policy))
            try:
                arrow_img = PhotoImage(file=action_map[action])
                self.arrows.append(arrow_img)   # prevent GC
                self.canvas.create_image(
                    col * UNIT + UNIT // 2,
                    row * UNIT + UNIT // 2,
                    image=arrow_img
                )
            except Exception:
                pass

    def draw_from_policy(self, policy_table):
        for row in range(HEIGHT):
            for col in range(WIDTH):
                self.draw_one_arrow(col, row, policy_table[row][col])

    # ------------------------------------------------------------------ #
    #  Print value table on canvas
    # ------------------------------------------------------------------ #

    def print_value_table(self, value_table):
        # Remove old value text
        for text in self.texts:
            self.canvas.delete(text)
        self.texts.clear()

        for row in range(HEIGHT):
            for col in range(WIDTH):
                val = round(value_table[row][col], 2)
                text = self.text_value(row, col, str(val))
                self.texts.append(text)

    # ------------------------------------------------------------------ #
    #  Button callbacks
    # ------------------------------------------------------------------ #

    def evaluate(self):
        self.evaluation_count += 1
        self.agent.policy_evaluation()
        self.print_value_table(self.agent.value_table)

    def improve(self):
        self.improvement_count += 1
        self.agent.policy_improvement()
        self.draw_from_policy(self.agent.policy_table)

    def move(self):
        self.move_by_policy()

    def reset(self):
        self.evaluation_count = 0
        self.improvement_count = 0
        # Reset agent tables
        self.agent.value_table = [[0.0] * WIDTH for _ in range(HEIGHT)]
        self.agent.policy_table = self.agent.init_policy()
        # Clear canvas items
        for text in self.texts:
            self.canvas.delete(text)
        self.texts.clear()
        for arrow in self.arrows:
            pass  # PhotoImage objects are already unreferenced
        self.arrows.clear()
        self.canvas.delete("arrow")
        # Move agent back to [0,0]
        self.canvas.coords(self.rectangle_item, UNIT // 2, UNIT // 2)

    def move_by_policy(self):
        """Animate the agent moving from (0,0) to the goal following the policy."""
        current_state = [0, 0]  # [col, row]
        max_steps = WIDTH * HEIGHT

        for _ in range(max_steps):
            row, col = current_state[1], current_state[0]
            policy = self.agent.policy_table[row][col]

            if max(policy) == 0:
                break

            action = policy.index(max(policy))

            next_state = self.agent.env.state_after_action(
                [current_state[1], current_state[0]], action
            )
            next_row, next_col = next_state

            # Move rectangle on canvas
            dx = (next_col - col) * UNIT
            dy = (next_row - row) * UNIT
            self.canvas.move(self.rectangle_item, dx, dy)
            self.update()
            time.sleep(0.1)

            current_state = [next_col, next_row]

            if [next_col, next_row] == GOAL:
                break


# ====================================================================== #
#  Environment
# ====================================================================== #

class Env:
    def __init__(self):
        self.height = HEIGHT
        self.width  = WIDTH
        self.reward = [[0.0] * WIDTH for _ in range(HEIGHT)]

        # Set goal reward
        self.reward[GOAL[1]][GOAL[0]] = 1.0

        # Set obstacle rewards
        for obs in OBSTACLES:
            col, row = obs
            self.reward[row][col] = -1.0

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.possible_actions = [0, 1, 2, 3]

    def get_reward(self, state, action):
        """Return the reward for taking action from state."""
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action):
        """Return the [row, col] after taking action from state."""
        row, col = state

        if action == 0:   # up
            row -= 1
        elif action == 1: # down
            row += 1
        elif action == 2: # left
            col -= 1
        elif action == 3: # right
            col += 1

        return self.check_boundary([row, col])

    def check_boundary(self, state):
        """Clamp state to within grid boundaries."""
        row, col = state
        row = max(0, min(row, self.height - 1))
        col = max(0, min(col, self.width  - 1))
        return [row, col]

    def get_all_states(self):
        """Return a list of all [row, col] states."""
        states = []
        for row in range(self.height):
            for col in range(self.width):
                states.append([row, col])
        return states
