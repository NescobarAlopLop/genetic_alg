import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
from inspect import signature
from shapes import Circle, Color
import random
from dna import DNA


class GeneticAlg:
    def __init__(self, dna_size, shape, img: np.ndarray, num_iter: int = 50):

        self.dna = DNA(dna_size, shape, img)
        self.current_cost = np.inf
        self.num_iter = num_iter
        self.result_image = img

    def run(self):
        try:
            for i, dna_mutation in enumerate(tqdm(self.get_mutation())):
                mutated_image = self.dna.grow_result(dna_mutation)
                mutation_cost = self.fitness(mutated_image, self.result_image)
                if mutation_cost < self.current_cost:
                    self.dna.apply(dna_mutation)
                    self.current_cost = mutation_cost
                    # showimage(mutated_image, self.result_image)
                if i % 100 == 0:
                    showimage(mutated_image, self.result_image)
            return self.dna.grow_result()
        except KeyboardInterrupt:
            showimage(mutated_image)

    def get_mutation(self):
        for _ in range(self.num_iter):
            yield self.dna.get_mutation()

    @staticmethod
    def fitness(arr1, arr2):
        return np.linalg.norm(arr1[:, :, 0] - arr2[:, :, 0]) + \
               np.linalg.norm(arr1[:, :, 1] - arr2[:, :, 1]) + \
               np.linalg.norm(arr1[:, :, 2] - arr2[:, :, 2])


class GeneratedImage:
    def __init__(self):
        self.shapes = []

    def combine_shapes(self):
        pass

    def normalize_shapes(self):
        pass

    def distance_to(self, arr):
        np.linalg.norm(self.normalize_shapes() - arr)


class Image:
    def __init__(self, path):
        self.image = imageio.imread(path)

    def plot_image(self):
        figure = plt.imshow(self.image)
        plt.show()

    def image_shape(self):
        return self.image.shape[:2]

    def __call__(self, *args, **kwargs):
        return self.image


# image = Image('circle.png')
# image.plot_image()


def showimage(img, img2: np.ndarray = None):
    plt.imshow(np.clip(img, 0, 255))
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.show()
    return img.shape


class SimulatedAnnealing:
    pass


if __name__ == '__main__':
    img = imageio.imread(
        '/home/ge/Documents/genetic_image_approximation/mona-lisa.jpg!PinterestLarge.jpg')
    # r = Circle('red', 180, 60, x=2, y=200)
    # g = Circle(1, 200, 50, x=60, y=200)
    # b = Circle(Color['blue'], 200, 50, x=80, y=200)

    # res = r + img
    # res = g + img
    # res = b + img
    # res = r + g
    # print('res', res.min(), res.max())
    alg = GeneticAlg(20, shape=Circle, img=img, num_iter=5000)
    she = alg.run()
    # fig = plt.figure()
    # ani = animation.FuncAnimation(fig, alg.run, interval=100)
    # plt.show()
    showimage(she, img)
    showimage(she)
