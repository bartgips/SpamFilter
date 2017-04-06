This is a repo for tracking my work on a spamfilter written in Python using Pandas and Sci-kit-learn

Data (spam and ham e-mail messages) can be found here:
http://www.aueb.gr/users/ion/data/enron-spam/

which is a database belonging to the following publication:

V. Metsis, I. Androutsopoulos and G. Paliouras, "Spam Filtering with 
Naive Bayes - Which Naive Bayes?". Proceedings of the 3rd Conference 
on Email and Anti-Spam (CEAS 2006), Mountain View, CA, USA, 2006.

To use:
preproc.py: to preprocess training data
train.py: to train a random forest using the preprocessed training data from preproc.py

classify.py [path]: call this script to classify a new e-mail (all email files in 'path') with trained model from

train.py (path to file can be used as argument to classify.py or be input to the consol when run without arguments)

Other files are scripts for debugging and generation of supporting figures
