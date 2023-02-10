# phone-call-speaker-emotion-recognition
End to end model to take a phone call recording .wav file as input, perform diarization to separate 2 voices and analyze emotion

* Importing required libraries: The first part of the code imports the required libraries such as numpy, pandas, keras, and scikit-learn.

* Augmentation methods: The code defines several data augmentation methods (noise, shift, stretch, pitch, dyn_change, speedNpitch) to artificially increase the size of the dataset and make the model more robust to variations in audio data.

* Reading the dataset: The script reads a CSV file (FINAL3.csv) that contains the labels, source, and file path of audio files. The file is read into a Pandas dataframe named 'ref'.

* Preprocessing the data: The script preprocesses the audio data and converts it into a form that can be used as input to the model. This includes transforming the audio into a feature representation, such as a spectrogram, and normalizing the data.

* Building and training the model: The code uses Keras to build a convolutional neural network (CNN) model and trains it on the preprocessed audio data. The script uses a validation set to monitor the performance of the model during training.

* Evaluating the model: The code uses the confusion matrix, accuracy score, and classification report metrics to evaluate the performance of the model on the test set.

* Model prediction: The script uses the trained model to make predictions on new audio files..
