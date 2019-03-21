import sys
from file_tools import *

def build_trimmed_dataset(input_filename, output_filename, minimum, maximum):
    '''
        Builds a trimmed version of the dataset with a minimum and maximum number
        of examples of each class.
    '''
    l = list_from_file(filename, "\n")
    if len(l) >= minimum:
        random.shuffle(l)
        with open(output_filename, "a+") as f:
            for i in range(0, maximum):
                f.write(l[i])

def main(argv):
    if len(argv) < 5:
        print("Usage: buid_trimmed_dataset.py savefile min max extension directory\n\
        savefile is the file to save the new dataset entries to.\n\
        min is the minimum number of examples to take from a file\n\
        max is the maximum number of examples. Use inf to take all of each example\n\
            though that does kind of defeat the purpose.\n\
        extension is the file extension of the datafiles.\n\
        directory is the location of the datafiles (leave blank for the present working directory).\n")
        exit()
    savefile = argv[1]
    minimum = int(argv[2])
    maximum = argv[3]
    if maximum == "inf":
        maximum = float('inf')
    else:
        maximum = int(maximum)
    extension = argv[4]
    if len(argv) == 6:
        directory = argv[5]
    else:
        directory = ""
    loop_over_files(build_trimmed_dataset, [savefile, minimum, maximum], directory, extension)

if __name__ == '__main__':
    main(sys.argv)