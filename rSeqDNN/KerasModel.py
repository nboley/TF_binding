import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Reshape,TimeDistributedDense, Permute)
from keras.layers.recurrent import LSTM,GRU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_yaml
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve, auc)

def encode_peaks_sequence_into_binary_array(peaks, fasta):
    # find the peak width
    pk_width = peaks[0].pk_width
    # make sure that the peaks are all the same width
    assert all(pk.pk_width == pk_width for pk in peaks)
    data = 0.25 * np.ones((len(peaks), 4, pk_width))
    for i, pk in enumerate(peaks):
        seq = fasta.fetch(pk.contig, pk.start, pk.stop)
        coded_seq = code_seq(seq)
        data[i] = coded_seq[0:4,:]
    return data

class KerasModel():
    def __init__(self, peaks_and_labels):
        self.seq_len = peaks_and_labels.max_peak_width
        numConv = 30
        convStack = 1
        convWidth = 4
        convHeight = 45
        dropoutRate = 0.2
        maxPoolSize = 50
        maxPoolStride = 20
        numConvOutputs = ((self.seq_len - convHeight) + 1)
        numMaxPoolOutputs = int(((numConvOutputs-maxPoolSize)/maxPoolStride)+1)
        gruHiddenVecSize = 35
        numFCNodes = 45
        numOutputNodes = 1
        # Define architecture     
        self.model = Sequential()
        self.model.add(Convolution2D(
            numConv, convStack, 
            convWidth, convHeight, activation="relu", init="he_normal"))
        self.model.add(Dropout(dropoutRate))
        self.model.add(MaxPooling2D(poolsize=(1,maxPoolSize),
                                    stride=(1,maxPoolStride)))
        self.model.add(Reshape(numConv,numMaxPoolOutputs))
        self.model.add(Permute((2,1)))
        # make the number of max pooling outputs the time dimension
        self.model.add(GRU(numConv,gruHiddenVecSize,return_sequences=True));
        self.model.add(TimeDistributedDense(gruHiddenVecSize,numFCNodes));
        self.model.add(Reshape(numFCNodes*numMaxPoolOutputs));
        self.model.add(Dense(numFCNodes*numMaxPoolOutputs,numOutputNodes,
                             activation='sigmoid'));
        sgd = SGD(lr=0.01,momentum=0.95,nesterov=True);
        self.model.compile(loss='binary_crossentropy', optimizer=sgd,
                           class_mode="binary");
    
    def get_features(self, coded_seqs):
        '''gets keras-formmated features and labels from data
        '''

        return np.reshape(coded_seqs, (len(coded_seqs), 1, 4, self.seq_len))
        
    def evaluate_rSeqDNN_model(self, X_validation, y_validation):
        '''evaluate model
        '''
        if len(np.shape(X_validation)) == 3: # reshape to 4D if 3D
            X_validation = self.get_features(X_validation)
        preds = self.model.predict_classes(X_validation)
        probs = self.model.predict_proba(X_validation)
        true_pos = y_validation == 1
        true_neg = y_validation == 0
        precision, recall, _ = precision_recall_curve(y_validation, probs)
        prc = np.array([recall,precision])
        auPRC = auc(recall, precision)
        auROC = roc_auc_score(y_validation, probs)
        classification_result = ClassificationResult(
            None, None, None, None, None, None,
            auROC, auPRC, 
            np.sum(preds[true_pos] == 1), np.sum(true_pos),
            np.sum(preds[true_neg] == 0), np.sum(true_neg)
        )

        return classification_result
    

    def train_rSeqDNN_model(self,
                            data,
                            genome_fasta,
                            out_filename_prefix,
                            numEpochs=5):
        # split into fitting and early stopping
        data_fitting, data_stopping = next(data.iter_train_validation_subsets())
        X_validation = self.get_features(
                encode_peaks_sequence_into_binary_array(
                        data_stopping.peaks, genome_fasta))
        y_validation = data_stopping.labels
        X_train = self.get_features(
                encode_peaks_sequence_into_binary_array(
                        data_fitting.peaks, genome_fasta))
        y_train = data_fitting.labels
        weights = {1, 1}
        batch_size = 1500
        # fit the model
        bestBalancedAcc = 0
        print("Training...")
        for epoch in xrange(numEpochs):
            self.model.fit(X_train, y_train,
                      validation_data=(X_validation, y_validation),
                      show_accuracy=True,
                      class_weight=weights,
                      batch_size=batch_size,
                      nb_epoch=1)
            res = self.evaluate_rSeqDNN_model(X_validation, y_validation)
            print res
            if (res.balanced_accuracy > bestBalancedAcc):
                print("highest balanced accuracy so far. Saving weights.")
                self.model.save_weights(out_filename_prefix + ".h5",
                                        overwrite=True)
                bestBalancedAcc = res.balanced_accuracy

        return self
