import os, sys

with open(sys.argv[1]) as fp:
    line_num = 0
    line = fp.readline()
    while line != '':
        if line.startswith(">"):
            if 'jolma' not in line:
                line = fp.readline()
                while line[0] != ">": 
                    line = fp.readline()
                continue
        print line,
        line = fp.readline()
