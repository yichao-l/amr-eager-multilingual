# amr-eager

AMR-EAGER [1] is a transition-based parser for Abstract Meaning Representation (http://amr.isi.edu/). The parser support English, Italian, Spanish, German and Chinese, with the following performance:

TODO: TABLE OF PERFORMANCE

# Installation

- Make sure you have Java 8
- Install Torch and torch packages dp, nngraph and optim (using luarocks, as explained here: http://torch.ch/docs/getting-started.html)
- Install the following python dependencies: numpy and pytorch (https://github.com/hughperkins/pytorch)
- Run ```./download.sh```
- For Spanish parsing, install FreeLing (tested 3.0 and 4.0) and set path in ```preprocessing_es.sh```  (https://github.com/TALP-UPC/FreeLing/releases)

# Run the parser with pretrained model

Note: the input file must contain sentences (one sentence for line), see ```contrib/sample-sentences.txt``` for example. All following commands should be run from the parser root directory.

Preprocessing:
```
./preprocessing.sh -s -l [en|it|de|es|zh] -f <sentence_file>
```

If not specified, the default language is English. You should get the output files in the same directory as the input files, with the prefix ```<sentences_file>``` and extensions ```.out``` and ```.sentences```.

```
python preprocessing.py -l [en|it|de|es|zh] -f <sentences_file>
```

You should get the output files in the same directory as the input files, with the prefix ```<sentences_file>``` and extensions ```.tokens.p```, ```.dependencies.p```.

Parsing:
```
python parser.py -l [en|it|de|es|zh] -f <file> -m <model_dir>
``` 
If you wish to have the list of all nodes and edges in a JAMR-like format, add option ```-n```. Without ```-m``` the parser uses the model provided in the directory ```ENGLISH```. For Spanish, you need to specify the model ```SPANISH```, for Italian ```ITALIAN```, for German ```GERMAN``` and for Chinese ```CHINESE```.

*Mac users: the pretrained models seem to have compatibility errors when running on Mac OS X.*

# Evaluation

We provide evaluation metrics to compare AMR graphs based on Smatch (http://amr.isi.edu/evaluation.html).
The script computes a set of metrics between AMR graphs in addition to the traditional Smatch code:

* Unlabeled: Smatch score computed on the predicted graphs after removing all edge labels
* No WSD. Smatch score while ignoring Propbank senses (e.g., duck-01 vs duck-02)
* Named Ent. F-score on the named entity recognition (:name roles)
* Wikification. F-score on the wikification (:wiki roles)
* Negations. F-score on the negation detection (:polarity roles)
* Concepts. F-score on the concept identification task
* Reentrancy. Smatch computed on reentrant edges only
* SRL. Smatch computed on :ARG-i roles only

The different metrics are detailed and explained in [1], which also uses them to evaluate several AMR parsers.
**(Some of the metrics were recently fixed and updated)**

```
cd amrevaluation
./evaluation.sh <file>.parsed <gold_amr_file>
```

To use the evaluation script with a different parser, provide the other parser's output as the first argument. 

# Train an English model
- Install JAMR aligner (https://github.com/jflanigan/jamr) and set path in ```preprocessing.sh```
- Preprocess training and validation sets:
  ```
  ./preprocessing.sh -f <amr_file>
  python preprocessing.py --amrs -f <amr_file>
  ```
  
- Run the oracle to generate the training data:
  ```
  python collect.py -t <training_file> -m <model_dir>
  python create_dataset.py -t <training_file> -v <validation_file> -m <model_dir>
  ```
  
- Train the three neural networks: 
  ```
  th nnets/actions.lua --model_dir <model_dir>
  th nnets/labels.lua --model_dir <model_dir>
  th nnets/reentrancies.lua --model_dir <model_dir>
  ```
  
  (use also --cuda if you want to use GPUs). 
 
- Finally, move the ```.dat``` models generated by Torch in ```<model_dir>/actions.dat```, ```<model_dir>/labels.dat``` and ```<model_dir>/reentrancies.dat```.
  
- To evaluate the performance of the neural networks run 
  ```
  th nnets/report.lua <model_dir>
  ```
- Note: If you used GPUs to train the models,you will need to uncomment the line ```require cunn``` from ```nnets/classify.lua```.

# Open-source code used:

- Smatch: http://amr.isi.edu/evaluation.html
- Tokenizer: https://github.com/redpony/cdec
- CoreNLP: http://stanfordnlp.github.io/CoreNLP/
- Tint: http://tint.fbk.eu
- FreeLing: https://github.com/TALP-UPC/FreeLing/releases

# References

[1] "An Incremental Parser for Abstract Meaning Representation", Marco Damonte, Shay B. Cohen and Giorgio Satta. Proceedings of EACL (2017).
