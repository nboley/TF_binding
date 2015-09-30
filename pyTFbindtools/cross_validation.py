from sklearn import cross_validation

TEST_CHRS = [1,2]
SINGLE_FOLD_VALIDATION_CHRS = range(3,5)
SINGLE_FOLD_TRAIN_CHRS = range(5, 23)

ClassificationResultData = namedtuple('ClassificationResult', [
    'is_cross_celltype',
    'sample_type', # should be validation or test
    'train_chromosomes',
    'train_samples', 

    'validation_chromosomes',
    'validation_samples', 

    'auROC', 'auPRC', 
    'num_true_positives', 'num_positives',
    'num_true_negatives', 'num_negatives'])

class ClassificationResult(ClassificationResultData):
    @property
    def positive_accuracy(self):
        return float(self.num_true_positives)/self.num_positives

    @property
    def negative_accuracy(self):
        return float(self.num_true_negatives)/self.num_negatives

    @property
    def balanced_accuracy(self):
        return (self.positive_accuracy + self.negative_accuracy)/2    
    
    def __str__(self):
        rv = []
        #rv.append(str(self.validation_samples).ljust(25))
        #rv.append(str(self.train_samples).ljust(15))
        rv.append("Balanced Accuracy: %.3f" % self.balanced_accuracy )
        rv.append("auROC: %.3f" % self.auROC)
        rv.append("auPRC: %.3f" % self.auPRC)
        rv.append("Positive Accuracy: %.3f (%i/%i)" % (
            self.positive_accuracy, self.num_true_positives,self.num_positives))
        rv.append("Negative Accuracy: %.3f (%i/%i)" % (
            self.negative_accuracy, self.num_true_negatives,self.num_negatives))
        return "\t".join(rv)

class ClassificationResults(list):
    def __str__(self):
        balanced_accuracies = [x.balanced_accuracy for x in self]    
        auROCs = [x.auROC for x in self]
        auRPCs = [x.auPRC for x in self]
        rv = []
        rv.append("Balanced Accuracies: %.3f (%.3f-%.3f)" % (
            sum(balanced_accuracies)/len(self),
            min(balanced_accuracies), max(balanced_accuracies)) )
        rv.append("auROC:               %.3f (%.3f-%.3f)" % (
            sum(auROCs)/len(self), min(auROCs), max(auROCs)))
        rv.append("auPRC:               %.3f (%.3f-%.3f)" % (
            sum(auRPCs)/len(self), min(auRPCs), max(auRPCs)))
        return "\n".join(rv)

def iter_train_validation_splits(sample_ids, contigs):
    # determine the training and validation sets
    if len(sample_ids) == 1:
        train_samples = sample_ids
        validation_samples = sample_ids
        all_sample_folds = [(train_samples, validation_samples),]
    else:
        all_sample_folds = []
        for sample in sample_ids:
            all_sample_folds.append(
                ([x for x in sample_ids if x != sample], [sample,]))
    # split the samples into validation and training
    non_test_chrs = sorted(
        set(contigs) - set("chr%i" % i for i in TEST_CHRS))
    all_chr_folds = list(cross_validation.KFold(
        len(non_test_chrs), n_folds=5))
    for sample_fold, chr_fold in itertools.product(
            all_sample_folds, all_chr_folds):
        train_samples, validation_samples = sample_fold
        train_chrs = [non_test_chrs[i] for i in chr_fold[0]]
        validation_chrs = [non_test_chrs[i] for i in chr_fold[1]]
        yield (
            (train_samples, train_chrs), 
            (validation_samples, validation_chrs))
    return
