#!/usr/local/bin/python3
# ./apply_code : apply a random code to an input file
#
# You don't have to do anything with this file -- it may just
# be useful for testing purposes
#
import random
import math
import copy 
import sys
import encode



if __name__== "__main__":
    if(len(sys.argv) != 3):
        raise Exception("usage: break_code.py input-file output-file")

    input = encode.read_clean_file(sys.argv[1])
    letters=list(range(ord('a'), ord('z')+1)) #returns a list of ascii values from a to z
    #print(letters)
    random.shuffle(letters)
    replace_table = dict(zip(map(chr, range(ord('a'), ord('z')+1)), map(chr, letters))) #creates a mapping dictionary from a to z like {a:'b', b:'a'}
    print(replace_table)
    print(input)
    rearrange_table = list(range(0,4)) 
    random.shuffle(rearrange_table) #prints shuffled version of [0,1,2,3]

    with open(sys.argv[2], "w") as file:
        print(encode.encode(input, replace_table, rearrange_table), file=file)

