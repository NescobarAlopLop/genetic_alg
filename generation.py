from heapq import heappush, nsmallest
from typing import List, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, ceil
from tqdm import tqdm

from dna import DNA
from genes import Circle


class Generation:
    def __init__(self, population_count, species_kind, image_to_esimtate,
                 num_iter: int = 5, genes_per_dna: int = 5):
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
        self.generation = [DNA(genes_in_dna=genes_per_dna,
                               geometric_shape=species_kind,
                               target_organism=image_to_esimtate)
                           for _ in range(population_count)]
        self.image_to_esimtate = image_to_esimtate
        self.num_iterations = num_iter
        self.population_count = population_count
        self.age = 0
        self.genes_per_dna = genes_per_dna

    def population_snapshot(self, iter: int = None, gen: List = None,
                            with_original=True):
        if gen is None:
            gen = self.generation
        num_plots_per_axis = ceil(sqrt(len(gen)))
        num_plots_per_axis = 3
        fig, axs = plt.subplots(num_plots_per_axis, num_plots_per_axis,
                                gridspec_kw={'wspace': 0.05, 'hspace': 0.05},
                                figsize=(4, 5), dpi=200, tight_layout=True)

        fig.suptitle(f'{iter}-th generation sample')
        gen_idx = 0

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

        plt.show()

    def evaluate_generation(self):
        h = []
        for dna in self.generation:
            heappush(h, dna)
        return h

    def get_n_best(self, n: int) -> List[DNA]:
        eval_gen = self.evaluate_generation()
        return nsmallest(n, eval_gen)

    def run(self):
        for i in tqdm(range(self.num_iterations)):
            n_best = self.get_n_best(2)
            self.new_generation(n_best)
            self.age += 1
            if i % 200 == 0:
                self.population_snapshot(i)
                best = self.get_n_best(1)[0]
                img = best.grow_result()
                if i % 400 == 0:
                    plt.imshow(img)
                plt.show()

    def new_generation(self, base_dnas: List):
        # base dnas transfer as is to next generation
        for i, dna in enumerate(base_dnas):
            self.generation[i] = dna

        # add mutated base dnas:
        for i, dna in enumerate(base_dnas):
            mut = dna.get_mutation(100)
            self.generation[i + len(base_dnas)].apply(mut)
            # dna.apply(dna.get_mutation(100))

        gene_pool = self.get_gene_pool(base_dnas)
        mutate = 0
        # create kids from halfs of dnas
        # and apply random to third of them
        for dna in self.generation[len(base_dnas) * 2:]:
            random_color_genes = np.random.choice(
                gene_pool.shape[0], self.genes_per_dna)
            random_shape_genes = np.random.choice(
                gene_pool.shape[0], self.genes_per_dna)
            dna.genes = np.hstack((
                gene_pool[random_color_genes, :3],
                gene_pool[random_shape_genes, 3:]))
            if mutate % 3 == 0:
                dna.apply(dna.get_mutation(90))
            mutate += 1
        return self.generation

    def __repr__(self):
        return f'<class Generation> age: {self.age}'

    @staticmethod
    def get_gene_pool(base_dnas: List[DNA]) -> np.ndarray:
        rv = []
        for dna in base_dnas:
            dna.age += 1
            for gene in dna.genes:
                rv.append(gene)
        return np.array(rv)


def fitness(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(arr1[:, :, 0] - arr2[:, :, 0]) + \
           np.linalg.norm(arr1[:, :, 1] - arr2[:, :, 1]) + \
           np.linalg.norm(arr1[:, :, 2] - arr2[:, :, 2])


if __name__ == '__main__':
    img = imageio.imread('circle.jpg')
    img = imageio.imread('black_base.jpeg')
    img = imageio.imread('mona-lisa.jpg!HalfHD.jpg')
    gen = Generation(population_count=9,
                     num_iter=2000,
                     species_kind=Circle,
                     image_to_esimtate=img,
                     genes_per_dna=20)
    gen.population_snapshot()
    gen.run()
    gen.population_snapshot()
    best = gen.get_n_best(1)[0]
    res = best.grow_result()
    plt.imshow(res)
    plt.show()
    plt.imshow(res)
    plt.imshow(img, alpha=0.5)
    plt.show()
