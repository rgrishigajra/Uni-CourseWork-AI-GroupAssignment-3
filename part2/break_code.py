#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: Rishabh Gajra Jay Madhu Milan Chheta rgajra jaymadhu michheta
#
# based on skeleton code by D. Crandall, 11/2019
#
# ./break_code.py : attack encryption
#


import random
import math
import copy 
import sys
import encode
import collections
import timeit
# put your code here!
# function to find probability of a string being english like
def find_probab(text,frequency_dictionary):
    probab=0
    if text[0] in frequency_dictionary:
        if ' ' in frequency_dictionary[text[0]]:
            probab=-math.log(frequency_dictionary[text[0]][' ']['probab'])
    for charIterator in range(1,len(text)):
        if text[charIterator] in frequency_dictionary:
            if text[charIterator-1] in frequency_dictionary[text[charIterator]]:
                probab=-math.log(frequency_dictionary[text[charIterator]][text[charIterator-1]]['probab']) + probab
    return probab

# function to decode the table with decrypt tables generated, based on encode function
def decode(str2, decrypt_rearrange_table, decrypt_replace_table):
    # pad with spaces to even multiple of rearrange table
    str2 +=  ' ' * (len(decrypt_rearrange_table)-(len(str2) %  len(decrypt_rearrange_table)))
    # and apply rearrange table"
    str3="".join(["".join([str2[decrypt_rearrange_table[j] + i] for j in range(0, len(decrypt_rearrange_table))]) for i in range(0, len(str2), len(decrypt_rearrange_table))])
    # apply replace table"
    str3 = str3.translate({ ord(i):ord(decrypt_replace_table[i]) for i in decrypt_replace_table })
    return str3

# swap elements in table with some probability
def modify_decryption_tables(decrypt_rearrange_table,decrypt_replace_table):
    # below variable chooses whether to swap elements in rearrangement table or replacement table with odds 4:325
    random_table_chooser=random.randint(1,331)
    if random_table_chooser <=6:
        # swapping two elements in the decrypt rearrangement table
        pos1=random.randint(0,3)
        pos2=random.randint(0,3)
        while pos1 == pos2 :
            pos2=random.randint(0,3)
        decrypt_rearrange_table[pos1], decrypt_rearrange_table[pos2] = decrypt_rearrange_table[pos2], decrypt_rearrange_table[pos1]
    else:
        # swapping two elements in the decrypt replacement table
        pos1=chr(random.randint(ord('a'),ord('z')))
        pos2=chr(random.randint(ord('a'),ord('z')))
        while pos1 == pos2 :
            pos2=chr(random.randint(ord('a'),ord('z')))
        decrypt_replace_table[pos1], decrypt_replace_table[pos2] = decrypt_replace_table[pos2], decrypt_replace_table[pos1]
    return [decrypt_rearrange_table,decrypt_replace_table]

def break_code(string, corpus):
    frequency_dictionary={}
    frequency_dictionary[corpus[0]]={}
    # calculate frequency of each character given the previous character in a nest dictionary where current key is outer key and previous character is inner key
    for charIterator in range(1,len(corpus)):
        if corpus[charIterator] not in frequency_dictionary:
            frequency_dictionary[corpus[charIterator]]={}
        if corpus[charIterator-1] in frequency_dictionary[corpus[charIterator]]:
                frequency_dictionary[corpus[charIterator]][corpus[charIterator-1]]['count']+=1
        else:
                frequency_dictionary[corpus[charIterator]][corpus[charIterator-1]]={'count':1, 'probab':0}
    # below loop calculates the probability of a letter given the previous letter occurred.
    for dictIterator1 in range(97,123):
        if chr(dictIterator1) in frequency_dictionary:
            for dictIterator2 in range(97,123):
                if chr(dictIterator2) in frequency_dictionary[chr(dictIterator1)]:
                    frequency_dictionary[chr(dictIterator1)][chr(dictIterator2)]['probab']=frequency_dictionary[chr(dictIterator1)][chr(dictIterator2)]['count']/len(string)
    # below loop calculates probabilties of first letter of words
    for dictIterator1 in range(97,123):
        if chr(dictIterator1) in frequency_dictionary:
            if ' ' in frequency_dictionary[chr(dictIterator1)]:
                frequency_dictionary[chr(dictIterator1)][" "]['probab']=frequency_dictionary[chr(dictIterator1)][' ']['count']/len(string)    
    # below loop makes all occurances of space as probability 1, since problem states you have to multiply probabilities of words
    for dictIterator1 in range(97,123):
        if chr(dictIterator1) in frequency_dictionary[' ']:
                frequency_dictionary[" "][chr(dictIterator1)]['probab']=1
    decrypted_answer=string
    min_probab=find_probab(string,frequency_dictionary)
    # generate the initial decrypt tables
    decrypt_rearrange_table = list(range(0,4)) 
    random.shuffle(decrypt_rearrange_table) #prints shuffled version of [0,1,2,3]
    decrypt_letters=list(range(ord('a'), ord('z')+1)) #returns a list of ascii values from a to z
    random.shuffle(decrypt_letters)
    decrypt_replace_table = dict(zip(map(chr, range(ord('a'), ord('z')+1)), map(chr, decrypt_letters))) #creates a mapping dictionary from a to z like {a:'b', b:'a'}
    # making first decryption
    decoded_string=decode(string,decrypt_rearrange_table,decrypt_replace_table)
    decoded_probab=find_probab(decoded_string,frequency_dictionary)
    if min_probab>decoded_probab:
            min_probab=decoded_probab
            decrypted_answer=decoded_string
    print(decrypted_answer,min_probab)  #print string and starting probab
    start_time=timeit.default_timer()
    while timeit.default_timer()-start_time < 601:
        # modifying the decryption tables
        [new_decrypt_rearrange_table,new_decrypt_replace_table] = modify_decryption_tables(copy.deepcopy(decrypt_rearrange_table),copy.deepcopy(decrypt_replace_table))
        # try to decode with new tables
        new_decoded_string = decode(string,new_decrypt_rearrange_table,new_decrypt_replace_table)
        new_decoded_probab = find_probab(new_decoded_string,frequency_dictionary)
        # if new tables are best yet, store them
        if min_probab>new_decoded_probab:
            min_probab=new_decoded_probab
            decrypted_answer=new_decoded_string
            # f= open("currentanswer.txt","w")
            # f.write(str(min_probab)+"\n")
            # f.write(decrypted_answer)
            # f.close()
            print(decrypted_answer,min_probab) #prints current best answer
        # if the new probability is better, switch tables
        if new_decoded_probab<decoded_probab :
            decrypt_rearrange_table = new_decrypt_rearrange_table
            decrypt_replace_table = new_decrypt_replace_table
            decoded_probab=new_decoded_probab
            decoded_string=new_decoded_string
        else :
            # if new probability is not better, switch to them with a small probability, otherwise continue while with new tables
            if decoded_probab>0:
                random_table_chooser=random.randint(0,int(new_decoded_probab))
                if random_table_chooser <= int(decoded_probab/50) :
                    decrypt_rearrange_table = new_decrypt_rearrange_table
                    decrypt_replace_table = new_decrypt_replace_table
                    decoded_probab=new_decoded_probab
                    decoded_string=new_decoded_string
            else:
                # when the probability of file goes negative
                random_table_chooser=random.randint(0,int(abs(decoded_probab)))
                if random_table_chooser <= int(abs(new_decoded_probab)/50) :
                    decrypt_rearrange_table = new_decrypt_rearrange_table
                    decrypt_replace_table = new_decrypt_replace_table
                    decoded_probab=new_decoded_probab
                    decoded_string=new_decoded_string
    return decrypted_answer


if __name__== "__main__":
    if(len(sys.argv) != 4):
        raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = encode.read_clean_file(sys.argv[1])
    corpus = encode.read_clean_file(sys.argv[2])
    decoded = break_code(encoded, corpus)

    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)

