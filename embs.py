#!/usr/bin/env python
#coding=utf-8

'''
Definition of OneHotEncoding class (for named entity sparse representation),
PretrainedEmbs (for pretrained embeddings) and 
RndInitLearnedEmbs (for random initialized ones).  # just indices starting from 1?
Embs puts everything together, using the same random seed.

@author: Marco Damonte (m.damonte@sms.ed.ac.uk)
@since: 03-10-16
'''

import re
import string
import random

class OneHotEncoding:
    def __onehot(self, index):
        onehot = [0]*(self.dim)
        onehot[index  - 1] = 1
        return onehot

    def  __init__(self, vocab):
        lines = open(vocab).readlines()
        self.dim = len(lines) + 3
        self.enc = {}
        for counter, line in enumerate(lines):
            self.enc[line.strip()] = self.__onehot(counter + 1)
        self.enc["<TOP>"] = self.__onehot(len(self.enc) + 1)
        self.enc["<NULL>"] = self.__onehot(len(self.enc) + 1)
        self.enc["<UNK>"] = self.__onehot(len(self.enc) + 1)
    def get(self, label):
        assert(label is not None)
        if label == "<TOP>":
            return self.enc["<TOP>"]
        if label.startswith("<NULL"):
            return self.enc["<NULL>"]
        if label in self.enc:
            return self.enc[label]
        return self.enc["<UNK>"]

class PretrainedEmbs:
    def __init__(self, generate, initializationFileIn, initializationFileOut, dim, unk, root, nullemb, prepr, punct):
        self.prepr = prepr # boolean
        self.indexes = {} # a dictionary from word to index.
        self.initialization = {}
        self.counter = 1
        self.dim = dim
        self.punct = punct
        self.nullemb = nullemb
        self.vecs = {}

        if generate:
            fw = open(initializationFileOut, "w")

        # Loop through all the word embeddings from the resources input.
        for line in open(initializationFileIn).readlines()[2:]: # first two lines are not actual embeddings (dims and </s>)
            v = line.split()

            # Extract word and embeddings seperatedly from initializationFileIn
            word = v[0]
            self.vecs[word] = " ".join(v[1:])
            
            # If need preprocessing.
            if self.prepr:
                word = self._preprocess(word)
            # If the word is already in the indexes diction, than continue with the next word.
            if word in self.indexes:
                continue
            # Enroll the word into the indexes word dictionary.
            self.indexes[word] = self.counter
            
            # Write the embeddings into the new output files, ',' seperated.
            if generate:
                fw.write(v[1])
                for i in v[2:]:
                    fw.write("," + str(i))
                fw.write("\n")
            self.counter += 1
        
        # Add "<UNK>", "<TOP>", "<NULL>" to the word->index dictionary, and write random embeddings to their corresponding output files enties.
        self.indexes["<UNK>"] = self.counter
        if generate:
            fw.write(str(unk[0]))
            for i in unk[1:]:
                fw.write("," + str(i))
            fw.write("\n")
        self.counter += 1

        self.indexes["<TOP>"] = self.counter
        if generate:
            fw.write(str(root[0]))
            for i in root[1:]:
                fw.write("," + str(i))
            fw.write("\n")
        self.counter += 1

        self.indexes["<NULL>"] = self.counter
        if generate:
            fw.write(str(nullemb[0]))
            for i in nullemb[1:]:
                fw.write("," + str(i))
            fw.write("\n")
        self.counter += 1

        if punct is not None:
            self.indexes["<PUNCT>"] = self.counter
            if generate:
                fw.write(str(punct[0]))
                for i in punct[1:]:
                    fw.write("," + str(i))
                fw.write("\n")
            self.counter += 1
        
    def get(self, word):
        assert(word is not None)
        if word == "<TOP>":
            return self.indexes["<TOP>"]
        if word.startswith("<NULL"):
            return self.indexes["<NULL>"]

        if self.prepr:
            word = self._preprocess(word)
        if self.punct is not None and word not in self.indexes and word in list(string.punctuation):
            return self.indexes["<PUNCT>"]
        elif word in self.indexes:
            return self.indexes[word]
        else:
            return self.indexes["<UNK>"]

    def _preprocess(self, word):
        if word.startswith('"') and word.endswith('"') and len(word) > 2:
            word = word[1:-1]
        reg = re.compile(".+-[0-9][0-9]")
        word = word.strip().lower()
        if reg.match(word) is not None:
            word = word.split("-")[0]
        if re.match("^[0-9]", word) is not None:
            word = word[0]
        word = word.replace("0","zero")
        word = word.replace("1","one")
        word = word.replace("2","two")
        word = word.replace("3","three")
        word = word.replace("4","four")
        word = word.replace("5","five")
        word = word.replace("6","six")
        word = word.replace("7","seven")
        word = word.replace("8","eight")
        word = word.replace("9","nine")
        return word
        
    def vocabSize(self):
        return self.counter - 1



class RndInitLearnedEmbs:
    def __init__(self, vocab):
        self.indexes = {}
        for counter, line in enumerate(open(vocab)):
            word = line.strip()
            self.indexes[word] = counter + 1
        self.indexes["<UNK>"] = len(self.indexes) + 1
        self.indexes["<TOP>"] = len(self.indexes) + 1
        self.indexes["<NULL>"] = len(self.indexes) + 1

    def get(self, label):
        assert(label is not None and label != "")
        if label == "<TOP>":
            return self.indexes["<TOP>"]
        if label.startswith("<NULL"):
            return self.indexes["<NULL>"]

        if label not in self.indexes:
            label = "<UNK>"
        return self.indexes[label]
        
    def vocabSize(self):
        return len(self.indexes)

class Embs:

    def _create_concept_vec(self, wordembs, conceptembs, propbank, wordvecs):
        fw = open(conceptembs, "w") 
        for line in open(wordembs):
            fw.write(line.strip() + "\n")
        for p in open(propbank):
            p2 = p.split("-")[0] 
            if p2 in wordvecs.indexes:
                fw.write(p.strip() + " " + wordvecs.vecs[p2] + "\n")
        fw.close()
            
    def __init__(self, resources_dir, model_dir, generate = False):
        random.seed(0)
        
	# Random float vectors with values [-0.01, 0.01)
        punct50 = [float(0.02*random.random())-0.01 for i in xrange(50)]
        punct100 = [float(0.02*random.random())-0.01 for i in xrange(100)]
        
        root10 = [float(0.02*random.random())-0.01 for i in xrange(10)]
        root50 = [float(0.02*random.random())-0.01 for i in xrange(50)]
        root100 = [float(0.02*random.random())-0.01 for i in xrange(100)]

        unk10 = [float(0.02*random.random())-0.01 for i in xrange(10)]
        unk50 = [float(0.02*random.random())-0.01 for i in xrange(50)]
        unk100 = [float(0.02*random.random())-0.01 for i in xrange(100)]

        null10 = [float(0.02*random.random())-0.01 for i in xrange(10)]
        null50 = [float(0.02*random.random())-0.01 for i in xrange(50)]

        # Assign each dep, pos a number.
        self.deps = RndInitLearnedEmbs(model_dir + "/dependencies.txt")
        self.pos =  RndInitLearnedEmbs(resources_dir + "/postags.txt")
        # Get the pretrained word embeddings.
        self.words = PretrainedEmbs(generate, resources_dir + "/wordvec50.txt", resources_dir + "/wordembs.txt", 50, unk50, root50, null50, True, punct50)
        self.nes = OneHotEncoding(resources_dir + "/namedentities.txt")
