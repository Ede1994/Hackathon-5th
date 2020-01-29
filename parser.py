# Copyright University College London 2019
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


def parser(path, split):
    with open(path, 'r') as file:
        for line in file:
            line = line.rstrip()

            split_array = line.split(split)

            if len(split_array) == 2:
                return split_array[1].split(',')
