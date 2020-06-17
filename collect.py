#!/usr/bin/env python
#coding=utf-8

'''
This script is used to collect resources such as the list of relation labels, 
dependency labels etc. These are usually computed on the training data, which
must be passed as an argument. The data must have been preprocessed with
preprocessing.sh and preprocessing.py.
Run as: python collect.py -t <training AMR file>

@author: Marco Damonte (m.damonte@sms.ed.ac.uk)
@since: 03-10-16
'''

import cPickle as pickle
from transition_system import TransitionSystem
from embs import Embs
from resources import Resources
import sys
import argparse
import re

def collect(prefix, language, model_dir):
    Resources.init_table(model_dir, True)

    print "Loading data.."
    # tokens, dependencies, alignments, relations for each sentece in the dataset.
    alltokens = pickle.load(open(prefix + ".tokens.p", "rb"))
    alldependencies = pickle.load(open(prefix + ".dependencies.p", "rb"))
    allalignments = pickle.load(open(prefix + ".alignments.p", "rb"))
    allrelations = pickle.load(open(prefix + ".relations.p", "rb"))

    print "Collecting relation labels.."
    # Store the set of all of the existed relation labels in the "relations.txt".
    seen_r = set()
    fw = open(model_dir + "/relations.txt","w")
    for relations in allrelations:
        for r in relations:
            if r[1] not in seen_r:
                fw.write(r[1] + "\n")
                seen_r.add(r[1])
    fw.close()

    print "Collecting dependency labels.."
    # Store the set of all of the existed dependencies labels in "dependencies.txt".
    seen_d = set()
    fw = open(model_dir + "/dependencies.txt","w")
    for dependencies in alldependencies:
        for d in dependencies:
            if d[1] not in seen_d:
                fw.write(d[1] + "\n")
                seen_d.add(d[1])
    fw.close()
    
    # Initialize the embeddings
    resources_dir = "resources_" + language
    embs = Embs(resources_dir, model_dir, True)
    
    counter = 0
    for tokens, dependencies, alignments, relations in zip(alltokens, alldependencies, allalignments, allrelations):
        counter += 1
        print "Sentence no: ", counter
        data = (tokens, dependencies, relations, alignments)
        t = TransitionSystem(embs, data, "COLLECT", language)

    Resources.store_table(model_dir)
    print "Done"

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--train", help="Training file to collect seen dependencies, AMR relations and other info", required = True)
    argparser.add_argument("-m", "--modeldir", help="Directory used to save the model being trained", required = True)
    argparser.add_argument("-l", "--lang", help="Language", default="en")

    try:
        args = argparser.parse_args()
    except:
        argparser.error("Invalid arguments")
        sys.exit(0)

    collect(args.train, args.lang, args.modeldir)
    print "Done"
