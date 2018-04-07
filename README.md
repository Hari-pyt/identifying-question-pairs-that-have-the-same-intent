# identifying-question-pairs-that-have-the-same-intent
TCS ExOP for Artificial Intelligence, (Assessment Task - CaseStudy)

# IMPLEMENTATION
1.	Place quora_duplicate_questions.csv and glove.840B.300d.txt file in the folder in which we run the program

             python main.py

2.	Now run the program Main.py, training takes place here, it will take 180 minutes to train,
Maximum validation accuracy and test accuracy will be displayed in command prompt and also question_pairs_weights.h5, word_embedding_matrix.npy and nb_words.json files will be generated in the same folder in which we run the program

             python predict.py

3.	Now we have required files to predict new data, the Predict.py will the the input as csv file and generate a output .csv file with predicted values

# DATA SET STUDY

1.  We have 404290 question pairs for both training and testing. Among 404290 questions pairs 149263 pairs have same intent (1) and remaining 255027 have different intent (0).
2.  93% questions have length 23 and 99% of the questions have at most length 31. 


# PREPROCESSING
1.	Building a tokenized word index
2.	Questions are converted into text to sequences
3.	Processing glove file
4.	Created a word embedding matrix of word index items
5.	Padded the sequences with zeros to a length of 25
6.	Partitioned the data in 90/10 train/test

# TRAINING
1.	Defined the model with 4 relu hidden layers and last layer is sigmoid layer.
•	Opted Relu as it is popular and it will not saturate easily
•	The last sigmoid layer will give the probability of the class of the question pairs
•	 Used binary cross-entropy as a loss function and Adam for optimization
2.	hidden layers are tested with 50,100,200 and 300 dimensions, among them 200 gives the best accuracy
3.	 Trained the network for 30 epochs with a further 90/10 train/test split
4.	saved the weights of the best validation accuracy

# PREDICTION
•	sigmoid gives the output probabilities of each question pair, ranging 0 to 1. 

# Implementation Framework
1.	python 3.5.2
2.	Packages are keras version (2.1.5), scikit-learn, pandas, numpy, h5py.
3.	After training saved the model weights, word embedding matrix files to make new predictions later.
4.	We can predict the values for new data using saved weights file
5.	Training takes approximately 360 seconds/epoch, 
6.  Trained on windows 7 machine consisting of 8GB Ram with i5 intel processor.

# References

1.	https://github.com/Smerity/keras_snli
2.	https://www.kaggle.com/c/quora-question-pairs/discussion/34325
3.	https://www.kaggle.com/c/quora-question-pairs
4.	https://towardsdatascience.com/natural-language-processing-with-quora-9737b40700c8
5.	http://nlp.stanford.edu/pubs/snli_paper.pdf
6.	https://explosion.ai/blog/quora-deep-text-pair-classification
7.	https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12
8.	https://deeplearning4j.org/glossary
9.	https://github.com/gmontamat/quora-question-pairs


