import numpy as np
from shapes import Color
import random
from inspect import signature
import matplotlib.pyplot as plt


def showimage(img, img2: np.ndarray = None):
    plt.imshow(np.clip(img, 0, 255))
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.show()
    return img.shape


class DNA:
    """
    self.dna is np.array, by columns:
    red, green, blue, ... shape parameters
    where red, green, blue is color intensity between 0 and 255
    for circle:
    red, green, blue, x_pos, y_pos, radius
    """
    def __init__(self, dna_size, geometric_shape, image_to_esimtate):
        self.generation = 1
        self.max_dna_size = dna_size
        self.current_dna_size = self.generation

        self.mutating_percentage = 100
        self.mutation_quantity = 20
        self.image_to_estimate = image_to_esimtate
        self.image_shape = image_to_esimtate.shape
        self.geometric_shape = geometric_shape
        self.avg_color_intensity = image_to_esimtate.sum(axis=2).mean()
        # sig = signature(geometric_shape.__init__)
        # num_params = len(sig.parameters) - 1
        self.dna = self.geometric_shape.get_random_params(self.image_to_estimate.shape)

    def extend_by(self, num_new_rows):
        for i in range(num_new_rows):
            np.vstack((self.dna, self.geometric_shape.get_random_params(self.image_to_estimate.shape)))
        return self.dna

    def get_mutation(self, percent_mutation: int = 70):
        self.generation += 1
        if self.generation % 100 == 0:
            self.dna = np.vstack((self.dna,
                                  self.geometric_shape.get_random_params(
                                      self.image_to_estimate.shape)))
        mutated_dna = self.dna.copy()
        num_shapes_to_mutate = mutated_dna.shape[0] * percent_mutation // 100
        rows_to_mutate = np.random.choice(range(mutated_dna.shape[0]),
                                          num_shapes_to_mutate, replace=False)
        for row in rows_to_mutate:
            mutated_dna[row, :] = \
                self.geometric_shape.mutate(
                    mutated_dna[row])
        return mutated_dna

    def get_next_generation(self, population_size=20):
        pass

    def apply(self, mutation):
        self.dna = mutation

    def grow_result(self, dna: np.array = None):
        if dna is None:
            dna = self.dna
        # rv = np.zeros(shape=self.image_shape)
        rv = np.zeros_like(self.image_to_estimate)
        shapes = np.apply_along_axis(lambda x: self.geometric_shape(
            **{'color': 0,
               'color_int': 200,
               'r': x[0],
               # 'r': x[0]//4,
               'g': x[1],
               'b': x[2],
               'x': x[3],
               'y': x[4],
               'radius': x[5],
               }), axis=1, arr=dna)

        for shape in shapes:
            rv = shape + rv
        rv[:, :3] = np.clip(rv[:, :3], a_min=0, a_max=255)
        return rv
