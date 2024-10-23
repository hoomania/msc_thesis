import numpy as np


class Lattice:

    def __init__(self, length: int, dim: int = 2):
        self.length = length
        self.dim = dim

    def generate(self, lattice_type: str = 'random') -> np.ndarray:
        if lattice_type == 'uniform':
            rnd_spin_value = 2 * np.random.randint(2) - 1
            if rnd_spin_value == 1:
                return np.ones((self.length, self.length), dtype=int)
            else:
                return np.zeros((self.length, self.length), dtype=int) - 1

        lattice = np.random.randint(2, size=(self.length, self.length))
        lattice = 2 * lattice - 1

        if self.dim == 1:
            return lattice.reshape(self.length * self.length)

        return lattice
