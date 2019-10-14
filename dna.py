import numpy as np
from shapes import Color
import random


class DNA:
    def __init__(self, dna_size, geometric_shape, image_to_esimtate):
        amount_of_red = image_to_esimtate[:, :, Color.red.value]
        amount_of_green = image_to_esimtate[:, :, Color.green.value]
        amount_of_blue = image_to_esimtate[:, :, Color.blue.value]

        self.image_shape = image_to_esimtate.shape

        avg_red_per_circle = amount_of_red // dna_size
        avg_blue_per_circle = amount_of_blue // dna_size
        avg_green_per_circle = amount_of_green // dna_size
        # sig = signature(geometric_shape.__init__)
        # num_params = len(sig.parameters) - 1
        # dtype = np.dtype([(x, np.uint8) for x in sig.parameters][1:])
        # zeros = np.random.randint(low=0, high=255)
        # self.dna = np.random.randint(low=0, high=255,
        # size=(dna_size, num_params), dtype=dtype)


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
