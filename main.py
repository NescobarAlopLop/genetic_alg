import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dna import DNA
from genes import Circle


class GeneticAlg:
    def __init__(self, dna_size, shape, img: np.ndarray, num_iter: int = 50):

        self.dna = DNA(geometric_shape=shape,
                       target_organism=img,
                       genes_in_dna=dna_size)
        self.current_cost = np.inf
        self.num_iter = num_iter
        self.result_image = img

    def run(self):
        try:
            for i, dna_mutation in enumerate(tqdm(self.get_mutation())):
                mutated_image = self.dna.grow_result(dna_mutation)
                mutation_cost = self.fitness(mutated_image)
                if mutation_cost < self.current_cost:
                    self.dna.apply(dna_mutation)
                    self.current_cost = mutation_cost
                    # showimage(mutated_image, self.result_image)
                if i % 100 == 0:
                    show_image(mutated_image, self.result_image)
            return self.dna.grow_result()
        except KeyboardInterrupt:
            show_image(mutated_image)

    def get_mutation(self):
        for _ in range(self.num_iter):
            yield self.dna.get_mutation()

    def fitness(self, arr1, arr2=None):
        if arr2 is None:
            arr2 = self.result_image
        return np.linalg.norm(arr1[:, :, 0] - arr2[:, :, 0]) + \
               np.linalg.norm(arr1[:, :, 1] - arr2[:, :, 1]) + \
               np.linalg.norm(arr1[:, :, 2] - arr2[:, :, 2])


def show_image(img, img2: np.ndarray = None):
    plt.imshow(np.clip(img, 0, 255))
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.show()
    return img.shape


if __name__ == '__main__':
    img = imageio.imread('mona-lisa.jpg!HalfHD.jpg')
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
    show_image(she, img)
    show_image(she)
