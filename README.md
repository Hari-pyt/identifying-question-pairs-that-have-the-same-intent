# identifying-question-pairs-that-have-the-same-intent
TCS ExOP for Artificial Intelligence, (Assessment Task - CaseStudy)

# implementation
1.	Place quora_duplicate_questions.csv and glove.840B.300d.txt file in the folder in which we run the program

             python main.py

2.	Now run the program Main.py, training takes place here, it will take 180 minutes to train,
Maximum validation accuracy and test accuracy will be displayed in command prompt and also question_pairs_weights.h5, word_embedding_matrix.npy and nb_words.json files will be generated in the same folder in which we run the program

             python predict.py

3.	Now we have required files to predict new data, the Predict.py will the the input as csv file and generate a output .csv file with predicted values

