# a3

# Part 1: Part-of-Speech tagging

**Formulation and code description:**
</br>
In this part our goal is to assign a part-of-speech to every word in the sentence using 3 types of Bayes Nets: Simple, HMM and MCMC. As given in the instructions we are only required to change the <i>pos_solver.py</i> file, hence it is the only file that has been modified. A new fucntion is defined called <i>calculate_probabilities</i> which calculates all the probabilities required for all the 3 models. This function is called in train function. Following probabilities are calculated:</br>
emission: p(W_i/t_i) = c(t_i,w_i)/c(t_i) </br>
transition: p(t_i+1,t_i) = c(t_i+1,t_i)/c(t_i) </br>
initial: p(t_i) = occurrce of t_i in first word of the sentence/length of data. </br>
Here t_i is POS tag and w_i is word. </br>

Starting with the **simple** model, where each observed variable is dependent only on the its own hidden variable, implementing this Bayes net was straight forward. To get the most probable tags for the words of sentence we find the tag which is has maximum probability amongst all the other tags associated with this word. </br>
-for each word </br>
&nbsp;&nbsp;&nbsp;-if w has t_1....t_k tags: </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-p(t_i/w) = c(w,t_i)/c(w,t_1) + ... + c(w, t_k) </br> 
Here c(w,t_i) is no. of times w/t_i appears in the corpus. </br>
If a word or tag is present in test set but not in train set, then that word is assigned the most occuring tag in the corpus. </br>

Next up is **HMM** which is solved using viterbi algorithm. In this model the observed variable is dependent on the it's hidden variable and there is also a dependency of hidden variable  on the hidden variable of the previous observed variable. Since viterbi uses the concept of dynamic programming we have maintained a list which holds a dicitonary, and the dictionary contains the probabilities for all the POS tags for a particular word (this makes column of word). Once, this so called matrix is calcualted we backtrack to get the list of most probable tags for the sentence. Viterbi algorithm has 3 parts.</br>
~ calculate probabilities for the first  column (or first word of the sentence) using initial probabilities and the emission probabilities.</br>
~ calculate probabilities for the rest of the column (or rest of the words in the sentence) using transition probabilities, emission probabilities and state probabilities.</br>
~ Find the maximum probability in the last column and backtrack to get the most probable tags for the sentence, append them into a list and return the list. </br>

</br>
For **MCMC**

</br>
**Other Dicussion:**
</br>
While there are no major design decision apart from the global dictionaries, there were many minor decisions or assumptions taken into consideration for different model. For example, in  the simple model we decided to assign a POS tag to a word which occurs the most no. of time in the corpus if the word is present in the test set but not in train set. Similarly for HMM, if the word and the POS tags are not in our dicitonaries (i.e. the trained data) then we assign a very small probability. The initial hurdle for us was to decide the structure of the code and how to train the data i.e., calculate the probabilities, once that was decided, the implementation was done according to the discussion done in class and ppts. To understand these models better for POS we referred to a few external sources like blogs and papers.

**Results:**
</br>
==> So far scored 2000 sentences with 29442 words.

|   | words correct  | sentences correct  | 
|---|---|---|
| 0. Ground Truth  | 100.00%  | 100.00%  | 
| 1. Simple  | 93.92%  | 47.45%  | 
| 2. HMM  | 95.06%  | 54.25%  | 
| 3. Complex  | 92.07%  | 40.52%  |


**

# **Part 2: Code breaking**

**
**

***#(1) a description of how you formulated the problem***

**

Since the Encryption is entirely random, the decoding I formulated was based on randomness as well, but in the opposite flow of the encoding function. I visualized this problem as a search where we have to maximize the likelihood of the text to be as English-like as possible. Since the probabilities were gonna be very small, I used negative logs instead and converted the maximization to the minimization of cost problem instead where cost are negative logs. The graph was something like this.

![](https://lh6.googleusercontent.com/k9otQZN73qFAr6Hs8RvB8TJkW-DIAaRxb3vmbSMLxOz8pX86rALwx-_RGHJhFKCOoBrB86qrMTljahfXX1G1pPlTH5igGo2tHVje3YbDaKIaFVpGmNonNjS2OQ3eepgzf_rsTDXW)

Here, the cost at A would be local minima, cost at B would be global minima. We need to return the **lowest value we have found yet as the answer after the ten-minute mark**. This could be that we never find B but only find A, ideally, B would be the answer that’s returned at the ten-minute mark. So we need to search along the graph for minima and store them as they come along. We use the metropolis hastings algorithm for that. Our search doesn’t always minimize the cost since this will cause it to be stuck at local minima if we encounter one before the global minima (the answer).

How the search works: 

 - At the start, we calculate the probability of the input string, by adding the probability of the first character the probability of the current letter given the previous letter for each letter of the word. 
 - Then the probability of each word added to find the probability of the entire string. This is the minima yet so we need to store it.
 -  Now we need to calculate two tables, rearrangement and replacement tables and decrypt text with them. This is our starting point. 
 - Calculating the probability of this string will give us the next point on the graph. We compare it with the minima and store the new text if it’s lower. 
 - Now we change the two tables a small bit and try decoding and calculating the probability again. If the probability increases it means we made the tables worse than what they initially were. 
 - If the probability is lesser, it’s in the right decision and we need to store these tables over our previous tables. 
 - Now, we can’t always ignore the worse side tables otherwise our search will stop in local minima. So with a small probability we give the freedom of going in the wrong direction to our search. We keep updating the minima as and when our search comes to them. 
 - The entire process is just changing our two tables one swap at a time to get as close to the original encrypting tables as possible.

**

***# (2) a brief description of how your program works***

**

*find_probab(text,frequency_dictionary):* 
	Returns a probability value for the text string using the frequency dictionary

*decode(str2, decrypt_rearrange_table, decrypt_replace_table):*
	Decodes the str2 string using decrypt tables and returns a string. This works in the opposite flow of the encode function by doing rearrangement first and then replacement.

*modify_decryption_tables(decrypt_rearrange_table,decrypt_replace_table):* 
	Modifies either of the tables by swapping two elements. The chance of rearrangement table to replacement table odds are 6:325. Since there’s 4 keys in the rearrangement table so 4C2 is the number of ways two keys can be selected at a time: 6. Similarly 26C2 for replacement table which is 326.

*break_code(string, corpus):* 
	Takes the string and corpus to return a decrypted answer in ten minutes. Calculates frequency in a nested dictionary such that outer keys are current text and inner keys are previous keys. a{b{count:1,probab:1}} means a occurring after b has already occurred happens once and the probability of that happening is one out of all the combinations. After probabilities are calculated we calculate initial encryption tables. We decrypt the string and then in a loop we modify tables, decrypt, check if these are better by comparing the probability values, store in minima if they are. This goes for ten minutes. We return minima at the end of the ten-minute mark as the final decrypted text.

  

**

***#(3) discussion of any problems you faced, any assumptions, simplifications, and/or design decisions you made.***

**

 - Initially, I thought of calculating the probabilities as the
   frequency of a current letter succeeding the previous letter divided
   by the frequency of the previous letter. Even though this made sense
   logically, the answers never came close and in fact when the inputs
   wherein perfect English got scores worse than that of random garbage
   text snippets. I changed this multiple times to finally settle to the
   length of the incoming string. These scaled-down probability values
   are enough to not make values too small to barely make any difference
   at any step to being too big so the jump in probability is too much
   and our program doesn’t make the jump thinking its wrong. 
   
 - Rather than calculating the probability of each word, I just add the probability of the current letter being space as 1 so the log of it is 0 and it ends up being a normal summation of all the word probabilities but in a single loop that runs for the length of the character. 
 - Also while deciding to go in the worse way where the new probability p(D’) is more in value than p(D), when I let the tables be selected even for worse swaps the code became too random. I noticed that the code would come close to the solution but just run away very easily. So i tried reducing the chance of this happening and made it more rare for thealgo to choose worse tables after swap instead forcing it to look for better solutions nearby, which was done by simply dividing the probability of choosing to do so ( p(D’)/p(D)) by a number like 50. This worked better in experimentation and the answer came closer so I choose to keep them. 
 - My values for probability did drop to negative once the text got more and more English like. Logically this doesn’t make sense but here since I had just scaled down my probabilities and not divided them by a proper total, this was expected. I had to make my code to work for both cases, where my probabilities worked in positive values and negative values. 
   
   *Test Outputs for files encrypted-text-1.txt is output1.txt,
   ecrypted-text-2.txt is output2.txt and so on. Each was run with 10
   minute limit. I also added a prop.txt file that was not encrypted to
   test how the code deals with such files, and it doesn’t change the
   input file at all since its already English like. Probab.txt is the
   dictionary of frequencies and the probabilities I am using to
   calculate the probability of the file.*

Example > encrypted-text-3
Input:
	 *i enpepn o echwxeh epiz lktpw fhnhfq cfe hzkc ndtewc onl cknlqlo kctfcnoe g egzcz atgcnk mnceatnpekouf nnzhcf utz pwgnf  pmewhzp qn fwfnfkh pige pny ekcxexcd guwec eocclkwp qnhfcqc hklpfkeup  lkcmy ch cc eooce w lkz cxcnnkkphce pnokcl cu ooocftwc pkoefckhwhf nqfh wccentdof cgae  lkckw fmc fthip accpx co kfckft  klegiy nwtitpohkocp tflock cwc pfp eytc pweo kmk fc mlnhotocxckk ucn lc ftzw ehnhfq cfchnlxcg  onfkfl  lkkmt c chexuco knkhoc ckuxn k et ocgxc lkmkk fc mlcvcok e ntpznok zgi nggzckpwzhe kcl ogcwc oachnllxe  oncpke k e cu dpcteptcwk e ncagn p ecpgncpgoc ouigsc tthwezecp e z nnaunfrgnpek geockotnop nikknpek clo ofm ezv jchf ginpqtk mnpk ncc lc omzk ecg lyl f peckknokeni hqppokfkcl  lkwkh fqklnl c iqtbckpwe z l ckikhee z xfcxoz ghk ecn lk lshiunkn n chocohatk cn o ee othcht  c*

Output :
	*o  union is more profound than marriage for it embodies the highest ideals of love fidelitw devotion sacrifice and familw n forming a marital union tyo people become something greater than once thew yere s some of the petitioners in these cases demonstrate marriage embodies a love that maw endure even past death t yould misunderstand these men and yomen to saw thew disrespect the idea of marriage heir plea is that thew do respect it respect it so deeplw that thew seex to find its fulfillment for themselves heir hope is not to be condemned to live in loneliness ekcluded from one of civilizations oldest institutions hew asx for equal dignitw in the ewes of the lay he onstitution grants them that right he judgment of the ourt of ppeals for the ikth ircuit is reversed t is so ordered*      

The above output was generated in ten minutes during our testing, which was pretty close to the answer, thus we didnt find a need to perform beam search or optimizing the algorithm.

<!----- Conversion time: 0.706 seconds.


Using this Markdown file:

1. Cut and paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β17
* Sun Dec 01 2019 18:25:31 GMT-0800 (PST)
* Source doc: https://docs.google.com/open?id=1VHThxp5QV7QiQXRFk3R5FOS0OWA5_-2donuziINBFeg
----->


**Part 3: Spam Classification **

**Naive Bayes classifier** is used to implement the spam classifier.

This program involves a **training part** where the likelihood probabilities are calculated based on the provided training data and a **testing part** where the program predicts where the file is spam or not.

_Methods used in the program:_

**Test() **- To test the data and predict whether the file is spam or not

**Train() **- To train on provided spam and non-spam file

**likelihood()** - To calculate likelihood probabilities using spam and non-spam files.

**trainSpam()** - To train on spam files.

**trainNotSpam()** -To train on non-spam files.

**Approach:**



*   We need to calculate **P(S/w1,w2,w3,…..wn)/ P(S_/w1,w2,w3,…..wn)** for each test file where _P(S/w1,w2,w3,…..wn)_ is probability of an email being a spam given the words _w1,w2,w3,…wn_ and _P(S_/w1,w2,w3,…..wn)_ is probability of an email not being a spam given the words _w1,w2,w3,…wn_.
*   For calculating these probabilities we need **P(w1,w2,w3,…..wn/S)*P(S)/P(w1,w2,w3,…..wn)** and **P(w1,w2,w3,…..wn/S_)*P(S_)/P(w1,w2,w3,…..wn)**.
*   But as we calculate odds ratio we only need**_ P(w1,w2,w3,…..wn/S), P(w1,w2,w3,…..wn/S_),_** _P(S)_ and _P(S_)._
*   Now, assumption is made that no word is dependent on other word thus we need _P(w1/S), P(w2/S)… P(wn/S)_ and _P(w1/S_), P(w2/S_)… P(wn/S_), _where P(wn/S) is probability of word w1 occurring in a spam file and _P(wn/S_)_ is probability of word _wn _occurring in non-spam file.

**Implementation:**



*   _P(S)_ and _P(S_) is _assumed as 0.5 as every mail has equal possibility of being a spam or not spam.
*   Frequency of words is calculated in **_trainSpam()_** and **_trainNotSpam()_** methods by going through every file in spam and non-spam data and breaking the file-text into token of words and saving the frequency of those words for every spam and non-spam files.
*   _P(wi/S) and P(wi/S_)_ is calculated in the **_likelihood()_** method during training process for every word tokenized in the training process. 

    **_P(wi/S) = frequency of the word in spam files/total number of words in spam files P(wi/S_)=  frequency of the word in non-spam files/total number of words in non-spam files_**

*   After the probabilities are calculated using training data, these probabilities are used while testing data for predicting whether a file is ‘_spam’_ or **‘_notspam’_**.
*   Every test file is converted to bag of words, and log of ratio of **_P(S/w1,w2…wn)/P(S_,w1,w2,..wn)_** is calculated for every word in bag of words whose probability is calculated with training data where,

    **_P(S/w1,w2…wn)/P(S_,w1,w2,..wn)= (P(w1/S)*P(w2/S)…*P(wn/S)*P(S))/ (P(w1/S_)*P(w2/S_)…*P(wn/S_)*P(S_))_**

*   If the total value is greater than 1 (meaning if the probability of that email being spam given words is greater than 0.5) than that email is labeled as **_spam_** else is labeled as **_notspam_**. The file and the resulting label are stored in a list and is written to an output file.

**Difficulties faced:**



*   Difficulty was faced while opening some of the files, due to **UniCodeError**.
*   Thus, the file is opened using **utf8** encoding and has attribute **errors=’ignore’**.
*   The **comma**, **semi-colon** and **equal to** is replaced from file text with **whitespace** to **tokenize** the words properly.
*   The word is converted to **lower case** before storing to avoid multiple copies of the same word in different cases.
*   A dictionary **‘vocab’** is maintained such that it contains all the unique words from training data and its frequency is maintained for spam and non-spam files.
*   The probabilities of test data are calculated using **log **to avoid working with extremely small values.

**Approaches tried:**



*   The likelihood probabilities are calculated as ->

    **frequency of the word in spam or non-spam files/total number of words in spam or non-spam files**

*   A different approach was tried originally where the likelihood probabilities were calculated as->

**Word appearances in spam or non-spam files/total number of that spam or non-spam**

**_Currently, the program predicts the spam file correctly 95% of the time._**


<!-- Docs to Markdown version 1.0β17 -->
