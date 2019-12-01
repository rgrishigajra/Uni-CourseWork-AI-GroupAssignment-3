# a3

# Part 1: Part-of-Speech tagging

**Formulation and code description:**
</br>
In this part our goal is to assign a part-of-speech to every word in the sentence using 3 types of Bayes Nets: Simple, HMM and MCMC. As given in the instructions we are only required to change the <i>pos_solver.py</i> file, hence it is the only file that is modified. A new fucntion is defined called <i>calculate_probabilities</i> which calculates all the probabilities required by all the 3 models. This function is called in train function.
Starting with the **simple** model, where each observed variable is dependent only on the its own hidden variable, implementing this Bayes net was straight forward. Assign a word the most probable POS from all the POS associated with that word. And if a word or tag is present in test set but not in train set, then that word is assigned the most occuring tag in the corpus. Accuracies achieved from this are 93. 92% (word) and 47.45% (sentence). 
Next up is **HMM** which is solved using viterbi algorithm. In this model the observed variable is dependent on the it's hidden variable and the hidden variable i.e. POS is dependent on the previous POS. Since viterbi uses the concept of dynamic programming we have maintained a list which holds a dicitonary which contains the probabilities for all the POS tags for a particular word (i.e column of word). Once, this so called matrix is calcualted we backtrack to get the list of most probable tags for the sentence.
For **MCMC**

**Other Dicussion:**
While there are no major design decision apart from the global dictionaries, there were many minor decisions or assumptions taken into consideration for different model. For example, in  the simple model we decided to assign a POS tag to a word which occurs the most no. of time in the corpus if the word is present in the test set but not in train set. Similarly for HMM, if the word and the POS tags are not in our dicitonaries (i.e. the trained data) then we assign a very small probability. The initial hurdle for us was to decide the structure of the code and how to train the data i.e., calculate the probabilities, once that was decided, the implementation was done according to the discussion done in class and ppts. To understand these models better for POS we referred to a few external sources like blogs and papers.

**Results:**
==> So far scored 2000 sentences with 29442 words.
|   | Words Correct   | Sentence Correct  | 
|---|---|---|
| 0. Ground Truth | 100.00%  | 100.00% |
| 1. Simple  | 93.92%  | 47.45%  |
| 2. HMM | 95.06%  | 54.25%  |
| 2. Complex | 18.60%  | 0.00%  |
