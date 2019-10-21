from heapq import heappush, nsmallest
from typing import List

import imageio
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, ceil

from dna import DNA
from genes import Circle


class Generation:
    def __init__(self, population_count, species_kind, image_to_esimtate,
                 num_iter: int=5):
        # generation is a number of DNAs
        # each DNA describes an image with chromosomes
        # so DNA is an image, chromosome is shape (circle) and generation is a
        # number of images

        # first generation starts as an array of random DNAs, which consist
        # of random shapes

        # each set of shapes (DNA) is combined/grown/drawn to create an image
        # images compared with original to get them fitness score (norm)
        # most fit DNAs have random children and produce new generation

        # concluding from youtube video it might be a good idea start with
        # small DNAs (less chromosomes / shapes ) and add complexity with time
        self.generation = [DNA(dna_len=1,
                               specie=species_kind,
                               image_to_estimate=image_to_esimtate)
                           for _ in range(population_count)]
        self.image_to_esimtate = image_to_esimtate
        self.num_iterations = num_iter
        self.population_count = population_count

    def population_snapshot(self, gen: List = None, with_original=True):
        if gen is None:
            gen = self.generation
        num_plots_per_axis = ceil(sqrt(len(gen)))
        fig, axs = plt.subplots(num_plots_per_axis, num_plots_per_axis,
                                gridspec_kw={'wspace': 0, 'hspace': 0.2})
        fig.suptitle('generation sample')
        gen_idx = 0
        try:
            for i in range(num_plots_per_axis):
                for j in range(num_plots_per_axis):
                    axs[i, j].set_title('dna: {}, fitness: {:.2f}'
                                        .format(gen_idx,
                                                gen[gen_idx].fitness_cost),
                                        fontsize=8)
                    axs[i, j].imshow(gen[gen_idx].grow_result())
                    if with_original:
                        axs[i, j].imshow(self.image_to_esimtate, alpha=0.5)
                    axs[i, j].axis('off')
                    gen_idx += 1
        except IndexError:
            pass
        plt.show()

    def evaluate_generation(self):
        h = []
        for dna in self.generation:
            heappush(h, (dna.fitness_cost, dna))
        return h

    def get_n_best(self, n: int) -> List:
        eval_gen = self.evaluate_generation()
        return nsmallest(n, eval_gen, key=lambda x: x[0])

    def run(self):
        for i in range(self.num_iterations):
            n_best = self.get_n_best(4)
            self.new_generation(n_best, self.population_count)
            # random cross mutation of 4 into self.population_count

    def new_generation(self, base_dnas: List, new_gen_size):
        a = np.array(base_dnas)
        new_gen = np.zeros(shape=(self.population_count, a.shape[1]))
        # TODO: duplicate dnas to increase population from 4 to n,
        # take into consideration each dna has more than one chromosomes
        for col in range(new_gen.shape[1]):
            new_gen[:, col] = np.random.choice(a[:, col], self.population_count)
        self.generation = new_gen
        # TODO: mixed up with generation definition
        return new_gen


def fitness(arr1, arr2):
    return np.linalg.norm(arr1[:, :, 0] - arr2[:, :, 0]) + \
           np.linalg.norm(arr1[:, :, 1] - arr2[:, :, 1]) + \
           np.linalg.norm(arr1[:, :, 2] - arr2[:, :, 2])


if __name__ == '__main__':
    img = imageio.imread('circle.jpg')
    gen = Generation(population_count=9,
                     species_kind=Circle,
                     image_to_esimtate=img)
    gen.population_snapshot()
    # gen_heap = gen.evaluate_generation()
    # n = 4
    # best_n = [heappop(gen_heap)[1] for _ in range(n)]
    # gen.population_snapshot(gen=best_n)
    gen.run()
