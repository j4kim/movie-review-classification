import shutil
import sys, os, random

ACCEPTED_TYPES = ('VER', 'NOM', 'ADJ', 'ADV')

def make_dir(folder, tag):
    '''
    Create a copy of a directory with the suffix given. If the directory already exists, adds a number at the end of the directory name
    :param folder: the path to the directory to copy
    :param tag: the suffix of the new folder
    :return: the name of the copy
    '''
    suffix = tag
    count = 1
    while True:
        try:
            os.mkdir(folder + '_' + suffix)
            return folder + '_' + suffix
        except FileExistsError:
            suffix = '{}_{}'.format(tag, count)
            count = count + 1

def sanitize(src, dst):
    '''
    From the tagged version of a test file, extract canonized verbs, adjectives
    and nouns and write them to a new file dst
    :param src: the source file to sanitize
    :param dst: the destination file where to write the sanitized version
    '''

    # open source file in read mode
    src_file = open(src, 'r', encoding='utf8')
    # open destination file in write mode
    dst_file = open(dst, 'w', encoding='utf8')

    # read and split each line of source file, write canonic word in destination file
    # if the type of the word is one of ACCEPTED_TYPES
    for line in src_file.readlines():
        try:
            raw, type, word = line.split()
            type = type.split(':')[0] # VER:pres -> VER
            if type in ACCEPTED_TYPES:
                dst_file.write(word + ' ')
        except:
            print('Unable to parse line: {}in file {}'.format(line, src))

    dst_file.close()
    src_file.close()


def separate(folder, test_ratio=0.2):
    '''
    Separate files from the subfolders 'pos' and 'neg' in two groups 'test' and 'train',
    sanitize their content and write them in a new folder
    :param folder: path to the folder containing subfolders pos and neg
    :param test_ratio: the percentage of files chosen for testing, the rest are for training
    '''

    # create an empty dir that will contain our separated files
    new_dir = make_dir(folder, 'prepared')

    # repeat for pos and neg folders
    for category in ['neg', 'pos']:
        # get a list of all files in the folder
        all = list(os.scandir(os.path.join(folder, category)))
        # randomize the list
        random.shuffle(all)
        # compute the amount of test test files
        test_size = int(len(all) * test_ratio)
        # create a small dict containing the train and test files
        files = {
            # select the n first files for testing
            'test':all[:test_size],
            # all the other files go for training
            'train':all[test_size:]
        }

        # repeat for training and testing groups
        for group in ['train', 'test']:
            # prepare the destination path
            dest = os.path.join(new_dir, group, category)
            # create the directories
            os.makedirs(dest)
            # process file content and then write to a new file in the destination folder
            for f in files[group]:
                sanitize(f.path, os.path.join(dest, f.name))


if __name__ == '__main__':
    try:
        folder = sys.argv[1]
    except:
        folder = 'tagged'

    separate(folder)