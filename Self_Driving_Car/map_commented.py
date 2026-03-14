# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0  # the total number of points in the last drawing
length = 0  # the length of the last drawing

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5, 3, 0.9)  # 5 sensors, 3 actions, gamma = 0.9
action2rotation = [0, 20, -20]  # action = 0 => no rotation, action = 1 => rotate 20 degrees, action = 2 => rotate -20 degrees
last_reward = 0  # initializing the last reward
scores = []  # initializing the mean score curve (sliding window of the rewards) with respect to time

# Initializing the map
first_update = True  # using this trick to initialize the map only once


def init():
    global sand, goal_x, goal_y, first_update
    sand = np.zeros((longueur, largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False


# Initializing the last distance
last_distance = 0


# Helper to safely read sand density around a sensor position
def _read_sensor(sensor_x, sensor_y, width, height):
    """Return sand density in a 20x20 window around (sensor_x, sensor_y), clamped to map bounds."""
    sx, sy = int(sensor_x), int(sensor_y)
    if sx > width - 10 or sx < 10 or sy > height - 10 or sy < 10:
        return 1.0  # out of bounds → full sand
    x0, x1 = max(sx - 10, 0), min(sx + 10, width)
    y0, y1 = max(sy - 10, 0), min(sy + 10, height)
    return int(np.sum(sand[x0:x1, y0:y1])) / 400.


# Creating the car class

class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos
        self.signal1 = _read_sensor(self.sensor1_x, self.sensor1_y, longueur, largeur)
        self.signal2 = _read_sensor(self.sensor2_x, self.sensor2_y, longueur, largeur)
        self.signal3 = _read_sensor(self.sensor3_x, self.sensor3_y, longueur, largeur)


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain, last_reward, scores, last_distance
        global goal_x, goal_y, longueur, largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[int(action)]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y

        last_distance = distance


# Painting for graphic interface

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[max(0, int(touch.x)), max(0, int(touch.y))] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.
            density = n_points / length
            touch.ud['line'].width = int(20 * density + 1)
            # Clamp indices to valid sand array range
            x0 = max(int(touch.x) - 10, 0)
            x1 = min(int(touch.x) + 10, sand.shape[0])
            y0 = max(int(touch.y) - 10, 0)
            y1 = min(int(touch.y) + 10, sand.shape[1])
            sand[x0:x1, y0:y1] = 1
            last_x = x
            last_y = y


# API and switches interface

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save', pos=(parent.width, 0))
        loadbtn = Button(text='load', pos=(2 * parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()


# Running the app
if __name__ == '__main__':
    CarApp().run()
