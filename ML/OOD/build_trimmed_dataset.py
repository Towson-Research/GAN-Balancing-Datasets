import sys
from file_tools import *
from separate_attacks import separate_attacks
import random

def build_trimmed_dataset(input_filename, output_filename, minimum, maximum, balance_normal = True):
    '''
        Builds a trimmed version of the dataset with a minimum and maximum number
        of examples of each class.

        If the balance_normal option is included, "Normal" traffic (if it exists in the 
        dataset) will be balanced to all other examples. (If there aren't enough examples
        of normal traffic all normal examples will be included.)

    '''
    all_attacks = separate_attacks(input_filename)
    count = 0
    with open(output_filename, "a+") as f:
        for attack in all_attacks:
            if len(attack) >= minimum:
                if attack[-1] != "normal":
                    random.shuffle(attack)
                    for i in range(0, maximum):
                        f.write(attack[i] + "\n")
                    count += 1
        # Write an appropriate number of normals
        if "normal" in all_attacks:
            normal = all_attacks["normal"]
            random.shuffle(normal)
            lim = max((balance_normal * maximum * (count -1 )) + maximum, len(normal))
            for i in range(0, lim):
                f.write(normal[i] + "\n")
        f.close()

def main(argv):
    if len(argv) < 5:
        print("Usage: buid_trimmed_dataset.py loadfile savefile min max -b\n\
        loadfile is the big dataset.\n\
        savefile is the file to save the new dataset entries to.\n\
        min is the minimum number of samples from a class to justify writing to the new set\n\
        max is the maximum number of examples from the randomized examples to keep in the new set.\n\
        include -b to balance the attack examples with normal examples")
        exit()
    loadfile = argv[1]
    savefile = argv[2]
    minimum = int(argv[3])
    maximum = argv[4]
    if maximum == "inf":
        maximum = float('inf')
    else:
        maximum = int(maximum)
    if len(argv) == 6 and argv[5] == "-b":
        balance = True
    else:
        balance = False
    build_trimmed_dataset(loadfile, savefile, minimum, maximum, balance)

if __name__ == '__main__':
    main(sys.argv)