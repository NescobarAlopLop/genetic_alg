from dna import DNA
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, ceil
import imageio
from shapes import Circle
from heapq import heappush, heappop
from typing import List


class Generation:
    def __init__(self, population_count, species_kind, image_to_esimtate):
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
        self.generation = [DNA(dna_size=1,
                               geometric_shape=species_kind,
                               image_to_esimtate=image_to_esimtate)
                           for _ in range(population_count)]
        self.image_to_esimtate = image_to_esimtate

    def population_snapshot(self, gen: List = None):
        if gen is None:
            gen = self.generation
        num_plots_per_axis = ceil(sqrt(len(gen)))
        fig, axs = plt.subplots(num_plots_per_axis, num_plots_per_axis)

        fig.suptitle('generation sample')
        gen_idx = 0
        try:
            for i in range(num_plots_per_axis):
                for j in range(num_plots_per_axis):
                    # axs[i, j].set_title('dna number: {}'.format(gen_idx))
                    axs[i, j].imshow(gen[gen_idx].grow_result())
                    axs[i, j].axis('off')
                    gen_idx += 1
        except IndexError:
            pass
        plt.show()

    def evaluate_generation(self):
        h = []
        for dna in self.generation:
            dna_mutation = dna.get_mutation()
            mutated_image = dna.grow_result(dna_mutation)
            dna_fitness = np.linalg.norm(self.image_to_esimtate, mutated_image)
            heappush(h, (dna_fitness, dna))
        return h


if __name__ == '__main__':
    img = imageio.imread('mona-lisa.jpg!HalfHD.jpg')
    gen = Generation(population_count=16, species_kind=Circle,
                     image_to_esimtate=img)
    gen.population_snapshot()
    gen_heap = gen.evaluate_generation()
    n = 5
    best_n = [heappop(gen_heap)[1] for _ in range(n)]
    gen.population_snapshot(gen=best_n)
