import time
import tkinter as tk
from PIL import ImageTk, Image

UNIT = 100
HEIGHT = 7
WIDTH = 7


class Env(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Q Learning')
        self.action_space = [0, 1, 2, 3]
        self.action_size = len(self.action_space)
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)

        # Draw grid lines
        for col in range(0, WIDTH * UNIT, UNIT):
            canvas.create_line(col, 0, col, HEIGHT * UNIT)
        for row in range(0, HEIGHT * UNIT, UNIT):
            canvas.create_line(0, row, WIDTH * UNIT, row)

        # Place images
        # Agent (rectangle)
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])

        # Obstacles (triangles)
        canvas.create_image(250, 50, image=self.shapes[1])
        canvas.create_image(550, 250, image=self.shapes[1])
        canvas.create_image(150, 450, image=self.shapes[1])
        canvas.create_image(350, 450, image=self.shapes[1])
        canvas.create_image(550, 150, image=self.shapes[1])
        canvas.create_image(350, 150, image=self.shapes[1])
        canvas.create_image(150, 350, image=self.shapes[1])

        # Destinations (circles)
        canvas.create_image(450, 150, image=self.shapes[2])
        canvas.create_image(250, 450, image=self.shapes[2])

        canvas.pack()
        return canvas

    def load_images(self):
        rectangle = ImageTk.PhotoImage(
            Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = ImageTk.PhotoImage(
            Image.open("../img/triangle.png").resize((65, 65)))
        circle = ImageTk.PhotoImage(
            Image.open("../img/circle.png").resize((65, 65)))
        return rectangle, triangle, circle

    def text_value(self, row, col, contents, action, font='Helvetica', size=7):
        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 42, 7
        else:
            origin_x, origin_y = 42, 77

        self.canvas.create_text(
            origin_x + UNIT * col,
            origin_y + UNIT * row,
            font=(font, size),
            text=contents)

    def print_value_all(self, q_table):
        for text in self.texts:
            self.canvas.delete(text)
        self.texts.clear()

        for state, values in q_table.items():
            state = eval(state)
            col = state[0]
            row = state[1]
            for action, value in enumerate(values):
                text = self.text_value(row, col, round(value, 2), action)
                self.texts.append(text)

    def coords_to_state(self, coords):
        x = int((coords[0] - 50) / UNIT)
        y = int((coords[1] - 50) / UNIT)
        return [x, y]

    def state_to_coords(self, state):
        x = int(state[0] * UNIT + 50)
        y = int(state[1] * UNIT + 50)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.3)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, 50 - x, 50 - y)
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = [0, 0]

        if action == 0:   # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        self.canvas.tag_raise(self.rectangle)
        next_state = self.coords_to_state(self.canvas.coords(self.rectangle))

        # Determine reward and done
        # Obstacle positions in state coords
        obstacles = [
            self.coords_to_state([250, 50]),
            self.coords_to_state([550, 250]),
            self.coords_to_state([150, 450]),
            self.coords_to_state([350, 450]),
            self.coords_to_state([550, 150]),
            self.coords_to_state([350, 150]),
            self.coords_to_state([150, 350]),
        ]
        # Destination positions in state coords
        destinations = [
            self.coords_to_state([450, 150]),
            self.coords_to_state([250, 450]),
        ]

        if next_state in destinations:
            reward = 100
            done = True
        elif next_state in obstacles:
            reward = -10
            done = False
        else:
            reward = 0
            done = False

        return next_state, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()
