###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import operator
import copy
import numpy as np


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    
    global_dic = {}
    tag_count = {}
    initial_prob = {}
    trans_prob = {}
    emission_prob = {}
    
    def calculate_probabilities(self, data):
        """Caculates the probabilities required to 
        solves all the 3 types of Bayes Nets"""
        for i in range(len(data)):
            sentences, pos_tags = data[i]
            
            #initial_prob: A dictionary which maintains probability that 
            #a given POS starts a sentence
            first_pos = pos_tags[0]
            if(first_pos not in self.initial_prob):
                self.initial_prob[first_pos] = 1
            else:
                fp_counter = self.initial_prob[first_pos]
                fp_counter += 1
                self.initial_prob[first_pos] = fp_counter
                
            for j in range(len(sentences)): #loop through all the words and their respective POS
                word = sentences[j]
                tag = pos_tags[j]         
                #calculating frequencies of two words occurring consecutively
                if(j+1 < len(sentences)):
                    if((pos_tags[j+1], tag) not in self.trans_prob):
                        self.trans_prob[(pos_tags[j+1], tag)] = 1
                    else:
                        self.trans_prob[(pos_tags[j+1], tag)] += 1
                #global_dic: A dictionary of Dictionaries to maintain
                #counts of different tags assigned to each words in the corpus.
                if(word not in self.global_dic):
                    self.global_dic[word] = {tag:1}
                else:
                    if(tag not in self.global_dic[word]):
                        self.global_dic[word].update({tag:1})
                    else:
                        tag_counter = self.global_dic[word][tag]
                        tag_counter += 1
                        self.global_dic[word][tag] = tag_counter
                #tag_count: A dictionary that maintains counts of occurrences
                #of each tag in the corpus.
                if(tag not in self.tag_count):
                    self.tag_count[tag] = 1
                else:
                    tag_countr = self.tag_count[tag]
                    tag_countr += 1
                    self.tag_count[tag] = tag_countr
        
        #final initial probabilities
        for tagg in self.initial_prob:
            self.initial_prob[tagg] /= len(data)
        
        #final transition probabilities
        for tups in self.trans_prob:
            self.trans_prob[tups] /= self.tag_count[tups[1]]
                
        #emission_prob: A dictionary which maintains emission probabilities
        self.emission_prob = copy.deepcopy(self.global_dic)
        for word in self.emission_prob:
            for tag in self.emission_prob[word]:
                self.emission_prob[word][tag] /= self.tag_count[tag]
    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        self.calculate_probabilities(data)

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        pos_tags = []
        for word in sentence:
            max_prob = 0
            curr_tag = ''
            if(word not in self.global_dic):
                #If the word in test set is not present in the train set,
                #assign the tag that appears maximum number of times in a corpus
                curr_tag = max(self.tag_count.items(), key=operator.itemgetter(1))[0]
            else:
                #get all the tags assigned to a word
                tags = self.global_dic[word]
                #calculate total assignment
                total = sum(tags.values())
                for s in tags:
                    p = tags[s]/total
                    if(p > max_prob):
                        max_prob = p
                        curr_tag = s
            pos_tags.append(curr_tag)
        return pos_tags

    def complex_mcmc(self, sentence):
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        return [ "noun" ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

