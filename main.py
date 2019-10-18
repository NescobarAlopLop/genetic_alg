import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
from inspect import signature
from shapes import Circle, Color
import random


class DNA:
    def __init__(self, dna_size, geometric_shape, estimated_image):
        amount_of_red = estimated_image[:, :, Color.red.value]
        amount_of_green = estimated_image[:, :, Color.green.value]
        amount_of_blue = estimated_image[:, :, Color.blue.value]
        avg_red_per_circle = amount_of_red // dna_size
        avg_blue_per_circle = amount_of_blue // dna_size
        avg_green_per_circle = amount_of_green // dna_size
        # sig = signature(geometric_shape.__init__)
        # num_params = len(sig.parameters) - 1
        # dtype = np.dtype([(x, np.uint8) for x in sig.parameters][1:])
        # zeros = np.random.randint(low=0, high=255)
        # self.dna = np.random.randint(low=0, high=255, size=(dna_size, num_params), dtype=dtype)
        self.image_shape = estimated_image.shape

        self.dna = []
        for i in range(dna_size):
            # sig = signature(geometric_shape.__init__)
            # args = {x: random.randint(1, 100) for x in sig.parameters}
            # args.__delitem__('self')
            args = geometric_shape.get_random_params()
            self.dna.append(geometric_shape(**args))

    def get_mutation(self):
        # mutated_dna = self.dna + \
        #               np.random.randint(-10, 10, size=self.dna.shape)
        # mutated_dna = []
        mutated_genes = random.choice(self.dna)
        # for shape in self.dna:
        #     mutated_dna.append(shape.apply_random_change())

        mutated_dna = self.dna.copy()
        for i in range(len(mutated_dna) // 20):
            gen_to_mutate = random.choice(mutated_dna)
            gen_to_mutate.apply_random_change()

        return mutated_dna

    def apply(self, mutation):
        self.dna = mutation

    def gen_result(self, dna: np.array = None):
        if dna is None:
            dna = self.dna
        rez = np.zeros(shape=self.image_shape)
        for shape in dna:
           rez = shape + rez
        return np.clip(rez, 0, 255)


class GeneticAlg:
    def __init__(self, dna_size, shape, img: np.ndarray, num_iter: int = 50):

        self.dna = DNA(dna_size, shape, img)
        self.current_cost = np.inf
        self.num_iter = num_iter
        self.result_image = img

    def run_loop(self):
        for i, dna_mutation in enumerate(tqdm(self.get_mutation())):
            mutated_image = self.dna.gen_result(dna_mutation)
            mutation_cost = self.cost(mutated_image, self.result_image)
            if mutation_cost < self.current_cost:
                self.dna.apply(dna_mutation)
                self.current_cost = mutation_cost

                # yield mutated_image + self.result_image
                showimage(mutated_image, self.result_image)

            if i % 200 == 0:
                # yield mutated_image + self.result_image
                showimage(mutated_image, self.result_image)
        return self.dna.gen_result()

    def run_generator(self):
        for i, dna_mutation in enumerate(tqdm(self.get_mutation())):
            mutated_image = self.dna.gen_result(dna_mutation)
            mutation_cost = self.cost(mutated_image, self.result_image)
            if mutation_cost < self.current_cost:
                self.dna.apply(dna_mutation)
                self.current_cost = mutation_cost

                yield mutated_image + self.result_image
                # showimage(mutated_image, self.result_image)

            if i % 200 == 0:
                yield mutated_image + self.result_image
                # showimage(mutated_image, self.result_image)
        return self.dna.gen_result()

    def get_mutation(self):
        for _ in range(self.num_iter):
            yield self.dna.get_mutation()

    @staticmethod
    def cost(arr1, arr2):
        return np.linalg.norm(arr1 - arr2)


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
    img = imageio.imread('mona-lisa.jpg!PinterestLarge.jpg')
    # r = Circle('red', 180, 60, x=2, y=200)
    # g = Circle(1, 200, 50, x=60, y=200)
    # b = Circle(Color['blue'], 200, 50, x=80, y=200)

    # res = r + img
    # res = g + img
    # res = b + img
    # res = r + g
    # print('res', res.min(), res.max())
    alg = GeneticAlg(300, shape=Circle, img=img, num_iter=2000)
    she = alg.run_loop()
    # fig = plt.figure()
    # ani = animation.FuncAnimation(fig, alg.run, interval=100)
    # plt.show()
    # showimage(she, img)
