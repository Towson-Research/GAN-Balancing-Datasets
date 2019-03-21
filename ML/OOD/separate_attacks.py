import sys
from file_tools import *

def separate_attacks(filename, separator = ","):
    '''
        Separates the attacks from the KDD99 dataset into individual files.
        The files are saved with the original data (all attacks) and unique data.
    '''
    with open(filename) as f:
        all_attacks = {}
        for line in f:
            line = line.strip("\n")
            line_list = line.split(separator)
            attack_type = line_list[-1]
            attack_type = attack_type[:-1]
            if attack_type not in all_attacks:
                all_attacks.update({attack_type : [line,]})
            else:
                all_attacks[attack_type].append(line)
        f.close()
    return all_attacks

def main(argv):
    if len(argv) < 2:
        print("Usage: separate_attacks.py inputfilename\n")
        exit()
    infile = argv[1]
    all_attacks = separate_attacks(infile)
    for attack in all_attacks:
        write_list_to_new_file(attack + ".data", all_attacks[attack])
        write_list_to_new_file(attack + "_unique" + ".data", list(set(all_attacks[attack])))

if __name__ == '__main__':
    main(sys.argv)