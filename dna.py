import numpy as np
from shapes import Color
import random


class DNA:
    """
    self.dna is np.array, by columns:
    red, green, blue, ... shape parameters
    where red, green, blue is color intensity between 0 and 255
    for circle:
    red, green, blue, x_pos, y_pos, radius
    """
    def __init__(self, dna_size, geometric_shape, image_to_esimtate):
        self.dna_size = dna_size
        amount_of_red = image_to_esimtate[:, :, Color.red.value].sum()
        amount_of_green = image_to_esimtate[:, :, Color.green.value].sum()
        amount_of_blue = image_to_esimtate[:, :, Color.blue.value].sum()

        self.image_shape = image_to_esimtate.shape
        self.geometric_shape = geometric_shape
        avg_red_per_circle = amount_of_red // dna_size
        avg_blue_per_circle = amount_of_blue // dna_size
        avg_green_per_circle = amount_of_green // dna_size

        # sig = signature(geometric_shape.__init__)
        # num_params = len(sig.parameters) - 1
        # TODO: custom dtype
        # dtype = np.dtype([(x, np.uint8) for x in sig.parameters][1:])
        dna_colors = np.random.randint(low=image_to_esimtate.min(),
                                       high=image_to_esimtate.max(),
                                       size=(dna_size, 3),
                                       dtype=np.uint8)

        # TODO: currently hard coded for circles, fix
        x_positions = np.random.randint(low=0,
                                        high=image_to_esimtate.shape[1],
                                        dtype=np.int,
                                        size=(dna_size, 1))
        y_positions = np.random.randint(low=0,
                                        high=image_to_esimtate.shape[0],
                                        dtype=np.int,
                                        size=(dna_size, 1))
        # max_rad = image_to_esimtate.shape[0] * image_to_esimtate.shape[1] // dna_size // 4
        max_rad = 20
        radii = np.random.randint(low=0,
                                  high=max_rad,
                                  dtype=np.int,
                                  size=(dna_size, 1))

        self.dna = np.hstack((dna_colors, x_positions, y_positions, radii))

    def get_mutation(self, percent_mutation: int = 5):
        # mutated_dna = self.dna + \
        #               np.random.randint(-10, 10, size=self.dna.shape)
        # mutated_dna = []
        mutated_genes = random.choice(self.dna)
        # for shape in self.dna:
        #     mutated_dna.append(shape.apply_random_change())
        num_shapes_to_mutate = self.dna_size * percent_mutation // 100
        rows_to_mutate = np.random.choice(np.arange(self.dna_size), num_shapes_to_mutate)
        mutation = np.random.randint(-4, 4, size=(rows_to_mutate.shape[0], self.dna.shape[1]))
        mutated_dna = np.copy(self.dna)
        mutated_dna[rows_to_mutate, :] = mutated_dna[rows_to_mutate, :] + mutation
        # mutated_dna = self.dna.copy()
        # for i in range(len(mutated_dna) // 20):
        #     gen_to_mutate = random.choice(mutated_dna)
        #     gen_to_mutate.apply_random_change()
        mutated_dna[:, :3] = np.clip(mutated_dna[:, :3], a_min=0, a_max=255)
        return mutated_dna

    def apply(self, mutation):
        self.dna = mutation

    def grow_result(self, dna: np.array = None):
        if dna is None:
            dna = self.dna
        rv = np.zeros(shape=self.image_shape)
        shapes = np.apply_along_axis(lambda x: self.geometric_shape(
            **{'color': 0, 'color_int': 200, 'radius': x[5], 'x': x[3],
               'y': x[4]}), 1, dna)

        for shape in shapes:
           rv = shape + rv
        rv[:, :3] = np.clip(rv[:, :3], a_min=0, a_max=255)
        return rv
