import sys

def file_to_setlist(filename, separator = "\n"):
    '''
        Converts file contents to a set, then to a list.
    '''
    with open(filename) as f:
        l = set()
        for line in f:
            line = line.strip("\n")
            if separator != "\n":
                line_entries = line.split(separator)
                for entry in line_entries:
                    l.add(entry)
            else:
                l.add(line)
        f.close()
    return list(l)

def write_list_to_new_file(outfile, l):
    with open(outfile, 'w') as f:
        for item in l:
            f.write(item + "\n")
        f.close()

def main(argv):
    if len(argv) < 3:
        print("Usage: list_to_set.py inputfilename outputfilename\n\
                Optionally add separator, e.g.: \n\
                list_to_set.py inputfilename outputfilename ;")
        exit()
    infile = argv[1]
    outfile = argv[2]
    if len(argv) == 4:
        separator = argv[3]
    l = file_to_setlist(infile)
    write_list_to_new_file(outfile, l)\

if __name__ == '__main__':
    main(sys.argv)