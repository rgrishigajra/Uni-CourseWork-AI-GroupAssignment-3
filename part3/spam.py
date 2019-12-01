import sys
import os
from math import log

#training on spam data
def trainSpam(vocab,pathSpam):
    wordCountSpam=0
    fileCountSpam=0

    #going through all given spam files
    for file in os.listdir(pathSpam):
        current = os.path.join(pathSpam, file)
        if os.path.isfile(current):
            fileCountSpam+=1
            with open(current, 'r', encoding='utf8', errors='ignore') as f:
                line = f.read().replace('\n', ' ').replace(',', ' ').replace('=', ' ').replace(';', ' ')
                for word in line.split():
                    word = word.lower()
                    #maintaing all the words in voca dictionary and their frequency in spam files
                    if word not in vocab:
                        vocab[word] = {'SpamCount': 1, 'NotSpamCount': 0}
                        wordCountSpam += 1
                    else:
                            vocab[word]['SpamCount'] += 1
                            wordCountSpam += 1

    return vocab,wordCountSpam,fileCountSpam

#training on notspam data
def trainNotSpam(vocab,pathNotSpam):
    fileCountNotSpam=0
    wordCountNotSpam=0

    #going through all notwspam files
    for file in os.listdir(pathNotSpam):
        current = os.path.join(pathNotSpam, file)
        if os.path.isfile(current):
            fileCountNotSpam+=1
            with open(current, 'r', encoding='utf8', errors='ignore') as f:
                line = f.read().replace('\n', ' ').replace(',', ' ').replace('=', ' ').replace(';', ' ')
                for word in line.split():
                    word = word.lower()

                    #maintaing all the words in voca dictionary and their frequency in notspam files
                    if word not in vocab:
                        vocab[word] = {'SpamCount': 0, 'NotSpamCount': 1}
                        wordCountNotSpam += 1
                    else:
                        vocab[word]['NotSpamCount'] += 1
                        wordCountNotSpam += 1

    return vocab,wordCountNotSpam,fileCountNotSpam


#calculating priors with the trained data
def priors(vocab,wordCountSpam,wordCountNotSpam):
    pSpam={}
    pNotSpam={}
    for word in vocab:
        #calculating priors of spam data
        if word not in pSpam:
            pSpam[word] = (vocab[word]['SpamCount']) / (wordCountSpam)

        #calculating priors of notspam data
        if word not in pNotSpam:
            pNotSpam[word] = (vocab[word]['NotSpamCount']) / (wordCountNotSpam)
    return pSpam,pNotSpam

#train data
def train(train_directory):
    pathSpam = train_directory + '/spam'
    pathNotSpam = train_directory + '/notspam'
    vocab={}
    #call training method for spam data
    vocab,wordCountSpam,fileCountSpam= trainSpam(vocab,pathSpam)

    #call training method for notspam data
    vocab,wordCountNotSpam,fileCountNotSpam = trainNotSpam(vocab,pathNotSpam)

    #calling method to calculate priors
    pSpam, pNotSpam = priors(vocab,wordCountSpam,wordCountNotSpam)

    return pSpam,pNotSpam,vocab,wordCountSpam,wordCountNotSpam,fileCountSpam,fileCountNotSpam


def groundTruth():
    truth = {}
    with open('test-groundtruth.txt', "r") as f:
        for line in f.readlines():
            filename, decision = line.split(' ')
            decision = decision.replace('\n', '')
            truth[filename] = decision
    return truth

#testing on given test data
def test(test_directory,vocab,wordCountSpam,wordCountNotSpam,fileCountSpam,fileCountNotSpam):
    pathTest = test_directory

    #probabilities of mail being spam or not spam
    spamProbab = 0.5
    notSpamProbab = 0.5

    right = 0
    wrong = 0
    outputFileSpam=[]
    outputFileNotSpam=[]
    #going through all files in test data
    for file in os.listdir(pathTest):
        current = os.path.join(pathTest, file)
        if os.path.isfile(current):
            with open(current, 'r', encoding='utf8', errors='ignore') as f:
                bag_of_words = []
                sum = log(spamProbab / notSpamProbab)
                line = f.read().replace('\n', ' ').replace(',', ' ').replace('=', ' ').replace(';', ' ')
                for word in line.split():
                    word = word.lower()
                    #maintaining the bag of words for current file
                    if word not in bag_of_words:
                        bag_of_words.append(word)

                        # #updating probabilities
                        # if word in pSpam:
                        #     vocab[word]['SpamCount'] += 1
                        #     wordCountSpam+=1
                        #     pSpam[word] = (vocab[word]['SpamCount'] ) / (wordCountSpam)
                        # if word in pNotSpam:
                        #     vocab[word]['NotSpamCount']+=1
                        #     wordCountNotSpam+=1
                        #     pNotSpam[word] = (vocab[word]['NotSpamCount'] ) / (wordCountNotSpam)

                #calculating fro each word in bag of words
                for word in bag_of_words:
                    if word in pSpam and pNotSpam:
                        if pSpam[word]!=0 and pNotSpam[word]!=0:
                            #calculating spam to notspam odds ratio for given words
                            sum += (log(pSpam[word])-log( pNotSpam[word]))
                #if odds ratio is greater than 1 then the file is labbeled as spam else notspam
                if sum > 1:
                    if truth[file] != 'spam':
                        wrong += 1
                        outputFileSpam += [str(file) + ' spam ' + str(sum) + ' wrong']
                    else:
                        right += 1
                        outputFileSpam += [str(file) + ' spam ' + str(sum) + ' right']
                    # outputFile += [str(file) + ' spam ' + str(sum)]
                else:

                    if truth[file] != 'notspam':
                        wrong += 1
                        outputFileNotSpam += [str(file) + ' notspam']
                    else:
                        right += 1
                        outputFileNotSpam += [str(file) + ' notspam']
                    # outputFile += [str(file) + ' notspam']
    print(right, wrong, (right / (right + wrong)) * 100)
    return outputFileSpam+outputFileNotSpam


#main function
if __name__ == '__main__':
    (train_directory, test_directory, output_file) = sys.argv[1:]

    #calling training methiod
    pSpam, pNotSpam,vocab,wordCountSpam,wordCountNotSpam,fileCountSpam,fileCountNotSpam = train(train_directory)

    truth = groundTruth()

    #calling test method
    outputFile = test(test_directory,vocab,wordCountSpam,wordCountNotSpam,fileCountSpam,fileCountNotSpam)
    output = './' + output_file

    #wrtiing output in output-file
    f = open(output, "w")
    for file in outputFile:
        f.write((file) + '\n')
    f.close()
