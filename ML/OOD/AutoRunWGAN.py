import numpy as np
from wgan import WGAN

def main():

    for i in range (0, 1000):
        a = np.random.randint(10, 41)
        b = np.random.randint(a, 70)
        c = np.random.randint(10, b)

        args = {
            'attack_type': "smurf",
            'max_epochs': 7000,
            'batch_size': 255,
            'sample_size': 500,
            'optimizer_learning_rate': 0.001,
            'generator_layers': [a, b, c]
        }
        for iter in range (0, 10):
            gan = WGAN(**args)
            gan.train()
            print("GAN finished with layers:")
            print(str([a, b, c]))


if __name__ == "__main__":
    main()