import numpy as np
from genes import Color
import random
from inspect import signature
import matplotlib.pyplot as plt


def showimage(img, img2: np.ndarray = None):
    plt.imshow(np.clip(img, 0, 255))
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.show()
    return img.shape


def fitness(arr1, arr2):
    return np.linalg.norm(arr1[:, :, 0] - arr2[:, :, 0]) + \
           np.linalg.norm(arr1[:, :, 1] - arr2[:, :, 1]) + \
           np.linalg.norm(arr1[:, :, 2] - arr2[:, :, 2])


class DNA:
    """
    self.dna is np.array, by columns:
    red, green, blue, ... shape parameters
    where red, green, blue is color intensity between 0 and 255
    for circle:
    red, green, blue, x_pos, y_pos, radius
    """
    def __init__(self, geometric_shape, target_organism: np.ndarray,
                 genes_in_dna: int = 1, genes: [None, np.ndarray] = None):

        self.age = 0
        self.max_genes = genes_in_dna

        self.mutating_percentage = 100
        self.mutation_quantity = 20
        self.image_to_estimate = target_organism
        self.target_shape = target_organism.shape
        self.gene_type = geometric_shape
        # sig = signature(geometric_shape.__init__)
        # num_params = len(sig.parameters) - 1
        if genes is None:
            self.genes = self.gene_type.generate_random(self.target_shape)
            if self.max_genes > 1:
                self.genes = self.extend_by(self.max_genes)
        else:
            self.genes = genes
        self.fitness_cost = fitness(self.grow_result(), self.image_to_estimate)

    def extend_by(self, num_new_rows: int) -> np.ndarray:
        for i in range(num_new_rows):
            self.genes = np.vstack([self.genes, self.gene_type.generate_random(self.target_shape)])
        return self.genes

    def get_mutation(self, percent_mutation: int = 70) -> np.ndarray:
        self.age += 1
        if self.age % 100 == 0:
            self.genes = np.vstack((self.genes,
                                    self.gene_type.generate_random(
                                      self.image_to_estimate.shape)))
        mutated_dna = self.genes.copy()
        num_shapes_to_mutate = mutated_dna.shape[0] * percent_mutation // 100
        rows_to_mutate = np.random.choice(range(mutated_dna.shape[0]),
                                          num_shapes_to_mutate, replace=False)
        for row in rows_to_mutate:
            mutated_dna[row, :] = \
                self.gene_type.mutate(
                    mutated_dna[row])
        return mutated_dna

    def apply(self, mutation):
        self.genes = mutation

    def grow_result(self, dna: [np.array, None] = None):
        if dna is None:
            dna = self.genes
        # rv = np.zeros(shape=self.image_shape)
        rv = np.zeros_like(self.image_to_estimate)
        shapes = np.apply_along_axis(lambda x: self.gene_type(
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

    def __repr__(self):
        return '<DNA: chromosomes: {}, ' \
            'fitness: {:2}>'.format(self.__len__(), self.fitness_cost)

    def __len__(self):
        return self.genes.shape[0]
