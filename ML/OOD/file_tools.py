from pathlib import Path

def loop_over_files(action, action_parameters = None, directory = "", file_extension = ""):
    '''
        action: A function to be performed on a batch of files.
        action_parameters: A list of the parameters for that function.

        Make sure any function that's going to use this takes the 
        filename as its first argument.
    '''
    pathlist = Path().glob(directory + "/" + file_extension)
    for path in pathlist:
        # because path is object not string
        filename = str(path)
        params = [filename] + action_parameters
        try:
            action(*params)
        except Exception as e:
            print(e)
    print("done")

def list_from_file(filename, separator, type_cast = None):
    '''
        Converts the contents of a separated file into a list.
        filename: the file to extract the contents of
        separator: e.g. , or ; for csv files
        type_cast: Optional to cast all members of the list before return
            Python will default file reads to str so you only need to
            supply this if you want something else, like int
    '''
    l = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            line = filter(None, map(str.strip, line.strip().split(separator)))
            if line:
                l += line

    # The * unpacks every element of an interator (i.e. the map object here) to a list.
    if type_cast:
        l = [*map(type_cast, l)]
    return l

def write_list_to_new_file(outfile, l):
    with open(outfile, 'w') as f:
        for item in l:
            f.write(item + "\n")
        f.close()
