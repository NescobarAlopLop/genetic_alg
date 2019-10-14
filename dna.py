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
        # sig = signature(geometric_shape.__init__)
        # num_params = len(sig.parameters) - 1
        # TODO: custom dtype
        # dtype = np.dtype([(x, np.uint8) for x in sig.parameters][1:])
        # dna_colors = np.random.randint(
        #     low=3,
        #     high=12,
        #     # low=image_to_esimtate.min(),
        #     #                            high=image_to_esimtate.max(),
        #                                size=(self.current_dna_size, 3),
        #                                # dtype=np.uint8
        # )
        dna_r = np.random.randint(
            low=self.image_to_estimate[:, :, 0].mean() - 30,
            high=self.image_to_estimate[:, :, 0].mean() + 30,
            size=(self.current_dna_size, 3),
            # dtype=np.uint8
        )
        dna_g = np.random.randint(
            low=self.image_to_estimate[:, :, 1].mean() - 30,
            high=self.image_to_estimate[:, :, 1].mean() + 30,
            size=(self.current_dna_size, 3),
            # dtype=np.uint8
        )
        dna_b = np.random.randint(
            low=max(self.image_to_estimate[:, :, 2].mean() - 30, 0),
            high=min(self.image_to_estimate[:, :, 2].mean() + 30, 200),
            size=(self.current_dna_size, 3),
            # dtype=np.uint8
        )
        # TODO: currently hard coded for circles, fix
        x_positions = np.random.randint(low=0,
                                        high=image_to_esimtate.shape[1],
                                        dtype=np.int,
                                        size=(self.current_dna_size, 1))
        y_positions = np.random.randint(low=0,
                                        high=image_to_esimtate.shape[0],
                                        dtype=np.int,
                                        size=(self.current_dna_size, 1))
        # max_rad = image_to_esimtate.shape[0] * image_to_esimtate.shape[1]
        # // dna_size // 4
        max_rad = 15
        radii = np.random.randint(low=12,
                                  high=max_rad,
                                  dtype=np.int,
                                  size=(self.current_dna_size, 1))

        # self.dna = np.hstack((dna_colors, x_positions, y_positions, radii))
        self.dna = np.hstack((dna_r, dna_g, dna_b, x_positions, y_positions, radii))


    def get_mutation(self, percent_mutation: int = 100):
        self.generation += 1
        # mutated_dna = self.dna + \
        #               np.random.randint(-10, 10, size=self.dna.shape)
        # mutated_dna = []
        mutated_genes = random.choice(self.dna)
        # for shape in self.dna:
        #     mutated_dna.append(shape.apply_random_change())
        num_shapes_to_mutate = self.current_dna_size * self.mutating_percentage // 100
        rows_to_mutate = np.random.choice(np.arange(self.current_dna_size), num_shapes_to_mutate)
        mutation = np.random.randint(-self.mutation_quantity, self.mutation_quantity, size=(rows_to_mutate.shape[0], self.dna.shape[1]))
        if self.generation % 100 == 0:
            self.current_dna_size += 1
            self.mutation_quantity = max(5, self.mutation_quantity - 1)
            self.mutating_percentage = max(30, self.mutating_percentage - 5)
            self.dna = np.vstack((self.dna, 30 * np.ones(shape=(1, self.dna.shape[1]))))
        mutated_dna = np.copy(self.dna)
        mutated_dna[rows_to_mutate, :] = mutated_dna[rows_to_mutate, :] + mutation
        # mutated_dna = self.dna.copy()
        # for i in range(len(mutated_dna) // 20):
        #     gen_to_mutate = random.choice(mutated_dna)
        #     gen_to_mutate.apply_random_change()
        mutated_dna[:, :3] = np.clip(mutated_dna[:, :3],
                                     a_min=self.image_to_estimate.min(),
                                     a_max=self.image_to_estimate.max())
        return mutated_dna

    def apply(self, mutation):
        self.dna = mutation

    def grow_result(self, dna: np.array = None):
        if dna is None:
            dna = self.dna
        rv = np.zeros(shape=self.image_shape)
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
