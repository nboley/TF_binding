import numpy as np
from abc import ABCMeta, abstractmethod
import os;
import sys;
scripts_dir = os.environ.get("UTIL_SCRIPTS_DIR");
if (scripts_dir is None):
raise Exception("Please set environment variable UTIL_SCRIPTS_DIR");
sys.path.insert(0,scripts_dir);
import pathSetter
from synthetic import synthetic;
from pyTFbindtools.sequence import code_seq
from pyTFbindtools.cross_validation import ClassificationResult

class SimulatedData:
	__metaclass__ = ABCMeta

	@abstractmethod
	def generate_pos_data(self):
		pass

	@abstractmethod
	def generate_neg_data(self):
		pass

def non_positional_fixed_embedder(loaded_motifs,motif_name,num_motifs):
	embedders=
	[
	    synthetic.RepeatedEmbedder(
	    synthetic.SubstringEmbedder(
	    substringGenerator=synthetic.PwmSamplerFromLoadedMotifs(
	    loadedMotifs=loaded_motifs,motifName=motif_name),
	    positionGenerator=synthetic.UniformPositionGenerator()),
	    quantityGenerator=synthetic.FixedQuantityGenerator(num_motifs))
	]
	return embedders

def positional_fixed_embedder(loaded_motifs,motif_name,num_motifs,central_bp):
	embedders=
	[
	    synthetic.RepeatedEmbedder(
	    synthetic.SubstringEmbedder(
	    substringGenerator=synthetic.PwmSamplerFromLoadedMotifs(
	    loadedMotifs=loaded_motifs,motifName=motif_name),
	    positionGenerator=synthetic.InsideCentralBp(central_bp)),
	    quantityGenerator=synthetic.FixedQuantityGenerator(num_motifs))
	]
	return embedders

def non_positional_poisson_embedder(loaded_motifs,motif_name,mean_motifs,max_motifs):
	embedders=
	[
	    synthetic.RepeatedEmbedder(
	    synthetic.SubstringEmbedder(
	    substringGenerator=synthetic.PwmSamplerFromLoadedMotifs(
	    loadedMotifs=loaded_motifs,motifName=motif_name),
	    positionGenerator=synthetic.UniformPositionGenerator()),
	    quantityGenerator=synthetic.MinMaxWrapper(
	    synthetic.PoissonQuantityGenerator(mean_motifs),
	    theMax=max_motifs))
	]
	return embedders

def positional_poisson_embedder(loaded_motifs,motif_name,mean_motifs,max_motifs,central_bp):
	embedders=
	[
	    synthetic.RepeatedEmbedder(
	    synthetic.SubstringEmbedder(
	    substringGenerator=synthetic.PwmSamplerFromLoadedMotifs(
	    loadedMotifs=loaded_motifs,motifName=motif_name),
	    positionGenerator=synthetic.InsideCentralBp(centered)),
	    quantityGenerator=synthetic.MinMaxWrapper(
	    synthetic.PoissonQuantityGenerator(mean_motifs),
	    theMax=max_motifs))
	]
	return embedders

class SimpleSimulations(SimulatedData):
'''
	- Randomly embedded once (single motif)
	- Randomly embedded once in centered region (positional)
	- Multiple motifs in pos. set, less motifs in neg set (density)
'''
	def __init__(self, pos_out, neg_out):
		self.pos_out = pos_out
		self.neg_out = neg_out

	def generate_pos_data(self, motif_name, num_motifs=1, mean_motifs=None, 
					  max_motifs=None, seq_len, num_seq, path_to_motifs, 
					  centered=None):

		loaded_motifs = synthetic.LoadedEncodeMotifs(path_to_motifs, pseudocountProb=0.001)
		if (centered is None): # non positional embedding
			if (mean_motifs == None and max_motifs == None): # fixed number of motifs generated
				embedder = non_positional_fixed_embedder()
				embed_in_background = synthetic.EmbedInABackground(
    				backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength=seq_len),
    				embedders=embedder)
    			else: # number of motifs sampled from poisson 
    				embedder = non_positional_poisson_embedder()
    				embed_in_background = synthetic.EmbedInABackground(
    				backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength=seq_len),
    				embedders=embedder)
		else: # positional embedding
			assert isinstance(centered, int), 'Must pass in integer for BPs to embed motif in.' 
			if (mean_motifs == None and max_motifs == None): # fixed number of motifs generated
				embedder = positional_fixed_embedder()
				embed_in_background = synthetic.EmbedInABackground(
    				backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength=seq_len),
    				embedders=embedder)
    			else: # number of motifs sampled from poisson 
    				embedder = positional_poisson_embedder()
    				embed_in_background = synthetic.EmbedInABackground(
    				backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength=seq_len),
    				embedders=embedder)
		
		sequence_set = synthetic.GenerateSequenceNTimes(embed_in_background, num_seq);
		synthetic.printSequences(self.pos_out, sequence_set);

	def generate_neg_data(self, motif_name, num_motifs=1, mean_motifs=None, 
					  max_motifs=None, seq_len, num_seq, path_to_motifs, 
					  centered=None, random=False):

		loaded_motifs = synthetic.LoadedEncodeMotifs(path_to_motifs, pseudocountProb=0.001)
		if (random is not False): # random sequence generation
			embedInBackground = synthetic.EmbedInABackground(
			backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength=30),
			embedders=[])
		elif (centered is None and random == False): # non positional embedding
			if (mean_motifs == None and max_motifs == None): # fixed number of motifs generated
				embedder = non_positional_fixed_embedder()
				embed_in_background = synthetic.EmbedInABackground(
    				backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength=seq_len),
    				embedders=embedder)
    			else: # number of motifs sampled from poisson 
    				embedder = non_positional_poisson_embedder()
    				embed_in_background = synthetic.EmbedInABackground(
				backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength=seq_len),
    				embedders=embedder)
		elif (centered is not None and random == False): # positional embedding
			assert isinstance(centered, int), 'Must pass in integer for BPs to embed motif in.' 
			if (mean_motifs == None and max_motifs == None): # fixed number of motifs generated
				embedder = positional_fixed_embedder()
				embed_in_background = synthetic.EmbedInABackground(
    				backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength=seq_len),
    				embedders=embedder)
    			else: # number of motifs sampled from poisson 
    				embedder = positional_poisson_embedder()
    				embed_in_background = synthetic.EmbedInABackground(
    				backgroundGenerator=synthetic.ZeroOrderBackgroundGenerator(seqLength=seq_len),
    				embedders=embedder)
	
		sequence_set = synthetic.GenerateSequenceNTimes(embed_in_background, num_seq);
		synthetic.printSequences(self.pos_out, sequence_set);

class GrammarSimulation(SimulatedData):

# fill in..

class TrainSimulations:

	def __init__(self, pos_seq_file, neg_seq_file):
		self.pos_seq_file = pos_seq_file
		self.neg_seq_file = neg_seq_file
		self.pos_matrix = []
		self.neg_matrix = []
		self.data = []
		self.labels = []

	def shuffle_in_unison_inplace(self, a, b):
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

	def convert_to_onehot(self):
		pos_seq = np.loadtxt(self.pos_seq_file,dtype="str")
		neg_seq = np.loadtxt(self.neg_seq_file,dtype="str")
		pos_matrix = np.zeros((len(pos_seq), 1, 5, len(pos_seq[0])))
		neg_matrix = np.zeros((len(neg_seq), 1, 5, len(neg_seq[0])))
	
		for i in range(len(pos_seq)):
			coded_seq = code_seq(pos_seq[i])
			coded_seq = np.reshape(coded_seq,(1,5,seq_len))
			pos_matrix[i,:,:,:] = coded_seq

		for i in range(len(neg_seq)):
			coded_seq = code_seq(neg_seq[i])
			coded_seq = np.reshape(coded_seq,(1,5,seq_len))
			neg_matrix[i,:,:,:] = coded_seq

		self.pos_matrix = pos_matrix
		self.neg_matrix = neg_matrix

	def join_datasets(self):
		labels1 = np.ones((len(self.pos_matrix),)).astype('int');
		labels2 = np.zeros((len(self.neg_matrix),)).astype('int');
		labels3 = np.concatenate((labels1,labels2),axis=0);
		combined = np.concatenate((self.pos_matrix,self.neg_matrix),axis=0);
		self.data = combined
		self.labels = labels3

	def split_train_test(self, train_split=0.75):
		data, labels = shuffle_in_unison_inplace(self.data, self.labels);
		split = int(len(data) * train_split);
		x_train = data[0:split];
		y_train = labels[0:split];
		x_test = data[split:len(data)];
		y_test = labels[split:len(data)];
		return x_train,x_test,y_train,y_test;

	def evaluate(self, model, X_validation, y_validation):
		preds = model.predict(X_validation)
		pred_probs = model.predict_proba(X_validation)        
		return ClassificationResult(y_validation, preds, pred_probs)

	def train(self, keras_model):
		convert_to_onehot()
		join_datasets()
		x_train,x_test,y_train,y_test = split_train_test()
		keras_model.compile(loss="binary_crossentropy",optimizer='adam')
		num_epochs = 20
		results = []
		for epoch in range(num_epochs):
			keras_model.fit(x_train,y_train,nb_epoch=1,batch_size=1000,show_accuracy=True)
			res = evaluate(keras_model, x_test, y_test)
			results.append(res) # contains ClassificationResult object for each epoch
		return results

