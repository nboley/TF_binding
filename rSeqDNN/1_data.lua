require 'torch'
require 'hdf5'
require 'mattorch'

print 'Loading dataset'

train_file = opt.training_file
valid_file = opt.validation_file
noutputs = 1


tr_size = opt.training_size
te_size = opt.validation_size


loaded = mattorch.load(train_file)
trainData = {
    data = loaded['trainxdata']:transpose(3,1),
    labels = loaded['traindata']:transpose(2,1),
    size = function() return tr_size end
}

loaded = mattorch.load(valid_file)
validData = {
    data = loaded['validxdata']:transpose(3,1),
    labels = loaded['validdata']:transpose(2,1),
    size = function() return te_size end
}
   
print 'Finished loading dataset'



