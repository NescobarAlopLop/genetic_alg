import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

import numpy as np
from skimage import draw


class Color(Enum):
    red = 0
    green = 1
    blue = 2


class Parameters(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Shape(ABC):

    def __init__(self, color: [int, str, Color], color_int, params: Dict):
        if isinstance(color, int):
            self.color = Color(color)
        elif isinstance(color, str):
            self.color = Color[color]
        elif isinstance(color, Color):
            self.color = color
        else:
            raise TypeError('color can be int, str or color enum'
                            'but cant be {}'.format(type(color)))
        if 0 <= color_int <= 255:
            self.color_intensity = np.uint8(color_int)
        else:
            raise ValueError("color has to be between 0 and 255")
        self.params = params

    def apply_random_change(self):
        self.color_intensity = self.color_intensity + random.randint(-5, 5)
        self.params = Parameters({k: v + random.randint(-5, 5) for k, v in self.params.items()})
        self.color_intensity = np.clip(
            self.color_intensity + random.randint(-5, 5),
            0, 255, dtype=np.uint8)

        # self.params.radius = np.clip(
        #     self.params.radius + random.randint(-4, 4),
        #     a_min=0, a_max=self.im
        # )
        # self.params.x
        # self.params.y
        return self

    @abstractmethod
    def draw_on(self, arr: np.array) -> np.array:
        pass

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.params}>'

    @abstractmethod
    def __add__(self, other):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass


class Circle(Shape):

    def __init__(self, color=None, color_int=None,
                 radius=None, x=None, y=None, r: int = 200, g: int = 200,
                 b: int = 200, clip: bool = True):
        if not color:
            color = Color(np.random.randint(0, 3))
        if not color_int:
            color_int = np.random.randint(1, 200, dtype=np.uint8)
        if not radius:
            radius = np.random.randint(1, 12, dtype=np.uint8)
        if not x:
            x = np.random.randint(1, 100, dtype=np.uint8)
        if not y:
            y = np.random.randint(1, 250, dtype=np.uint8)
        if clip and radius < 4:
            radius = 5
        if clip and radius > 60:
            radius = 50
        if radius <= 0:
            raise ValueError("Circle radius has to be greater than zero")
        self.r = r
        self.g = g
        self.b = b
        self.color = color
        self.color_int = color_int
        self.x = x
        self.y = y
        self.radius = radius
        params = Parameters({'radius': radius, 'x': x, 'y': y})
        Shape.__init__(self, color, color_int, params)
        # if not color:
        #     color = Color(np.random.randint(0, 3))
        # if not color_int:
        #     color_int = np.random.randint(1, 200, dtype=np.uint8)
        # if not radius:
        #     radius = np.random.randint(1, 12, dtype=np.uint8)
        # if not x:
        #     x = np.random.randint(1, 100, dtype=np.uint8)
        # if not y:
        #     y = np.random.randint(1, 250, dtype=np.uint8)
        # if round_up and radius <= 0:
        #     radius = 1
        # if radius <= 0:
        #     raise ValueError("Circle radius has to be greater than zero")
        # params = Parameters({'radius': radius, 'x': x, 'y': y})
        # Shape.__init__(self, color, color_int, params)

    def draw_on(self, arr: np.array) -> np.array:
        rr, cc = draw.circle(r=self.y,
                             c=self.x,
                             radius=self.radius,
                             shape=arr.shape)
        arr[rr, cc, 0] = self.r
        arr[rr, cc, 1] = self.g
        arr[rr, cc, 2] = self.b

        # rr, cc = draw.circle(r=self.params.y,
        #                      c=self.params.x,
        #                      radius=self.params.radius + 1,
        #                      shape=arr.shape)
        # arr[rr, cc, self.color.value] = self.color_intensity
        return arr

    @property
    def shape(self):
        x_dim = self.params.x + self.params.radius
        y_dim = self.params.y + self.params.radius
        return x_dim, y_dim

    def __add__(self, other: ['Circle', np.ndarray]):
        if isinstance(other, np.ndarray):
            return self.draw_on(other)
        y = max(self.shape[0], other.shape[0])
        x = max(self.shape[1], other.shape[1])
        empty = np.zeros((x, y, 3), dtype=np.int)
        me = self.draw_on(empty)
        return np.clip(other.draw_on(me), 0, 255, dtype=np.uint8)

    @staticmethod
    def get_random_params():
        return {
            'x': random.randint(0, 250),
            'y': random.randint(0, 450),
            'color_int': random.randint(100, 180),
            'color': random.randint(0, 2),
            'radius': random.randint(20, 70)
        }
