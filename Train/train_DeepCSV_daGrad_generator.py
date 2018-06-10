
from training_base import training_base
from Losses import loss_NLL

import pdb
import tensorflow as tf
from keras import backend as K
from datetime import datetime
import pickle
import numpy as np
import h5py
import os
from pdb import set_trace


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)  # ,per_process_gpu_memory_fraction=0.1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

from Helpers import shuffle_in_unison, get_class_weights
from DataCollection import DataCollection

print( "start training at " +str(datetime.now()))

### Gather the samples together

print('load sample files ...')

f = h5py.File(
    '/storage/c/dwalter/Data/Pretraining/Pretraining_Thu_174149_full2016_HDF5/ntuples.hdf5',
    'r')
pretrain_y = f['y'][()]
pretrain_x = f['x'][()]
f.close()

dc = DataCollection(infile='/storage/c/dwalter/Data/ttbar_Semilep/converted_TrainData_deepCSV_raw/Data/dataCollection.dc')
data_y = dc.getAllLabels()
data_x = dc.getAllFeatures()
data_w = dc.getAllWeights()
data_y2 = np.ones(len(data_y))

mc_y = []
mc_x = []
mc_w = []

def fill_mc(filename, weight = 1.):
    dc = DataCollection(infile=filename)
    mc_y.append(dc.getAllLabels())
    mc_x.append(dc.getAllFeatures())
    mc_w.append(dc.getAllWeights()*weight)

    #f = h5py.File(
    #filename,
    #'r')
    #mc_y.append(f['y'][()])
    #mc_x.append(f['x'][()])
    #mc_w.append(f['w'][()]*weight)
    #f.close()



#fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbar_Dilep/tt/HDF5/ntuples.hdf5', 831760*1./(77081156 + 77867738))
#fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbar_Dilep/dy50/HDF5/ntuples.hdf5', 6025200*1./81781052)
#fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbar_Dilep/dy10to50/HDF5/ntuples.hdf5', 22635100*1./47946519)
#fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbar_Dilep/ww/HDF5/ntuples.hdf5', 118700*1./994012)
#fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbar_Dilep/wz/HDF5/ntuples.hdf5', 44900*1./1000000)
#fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbar_Dilep/zz/HDF5/ntuples.hdf5', 15400*1./998034)
#fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbar_Dilep/wantit/HDF5/ntuples.hdf5', 35600*1./6933094)
#fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbar_Dilep/wt/HDF5/ntuples.hdf5', 35600*1./6952830)
#fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbar_Dilep/wjets/HDF5/ntuples.hdf5', 61526700*1./16497031)

basedir = '/storage/c/dwalter/Data/ttbar_Semilep/converted_TrainData_deepCSV_raw_undefinedIncluded/'
fill_mc(basedir+'TT/dataCollection.dc',831760./(77081156 + 77867738))
fill_mc(basedir+'DY10to50/dataCollection.dc',22635100./47946519)
fill_mc(basedir+'DY50/dataCollection.dc',6025200./81781052)
fill_mc(basedir+'Antit_tchannel/dataCollection.dc',80950./38811017)
fill_mc(basedir+'T_tchannel/dataCollection.dc',136020./67240808)
fill_mc(basedir+'Wantit/dataCollection.dc',35600./6933094)
fill_mc(basedir+'Wt/dataCollection.dc',35600./6952830)
fill_mc(basedir+'W2Jets/dataCollection.dc',3161000./29878415)
fill_mc(basedir+'W3Jets/dataCollection.dc',948200./19798117)
fill_mc(basedir+'W4Jets/dataCollection.dc',494600./9170576)


mc_y = np.concatenate(mc_y)
mc_x = np.concatenate(mc_x)
mc_w = np.concatenate(mc_w)
mc_y2 = np.zeros(len(mc_y))


norm = np.sum(data_w)/np.sum(mc_w)
print("norm = " +str(norm))
mc_w = mc_w * norm

da_x = np.vstack((data_x,mc_x))
da_y = np.vstack((data_y,mc_y))
da_y2 = np.expand_dims(np.append(data_y2,mc_y2),axis=1)
da_w = np.vstack((data_w,mc_w))

#pdb.set_trace()

print("shuffle ...")
da_x, da_y, da_y2, da_w = shuffle_in_unison((da_x,da_y,da_y2,da_w))

#pdb.set_trace()

#class_weight=get_class_weights(pretrain_y)

#relative ratio of jets of each flavour for b:c:udsg is set to 2:1:4 ; weight for bb is 1/178
#class_weight[0] *=2
#class_weight[1] /= class_weight[1]
#class_weight[3] *= 4
#print("use class weights ", class_weight)

print("got all the samples!")


### Training part
# also does all the parsing
train= training_base(testrun=False)
print 'Inited'




# print('shuffle samples in union')
# pretrain_y, pretrain_x, pretrain_w = shuffle_in_unison([pretrain_y, pretrain_x, pretrain_w])

#train.loadModel('/local/scratch/ssd1/dwalter/data/Ntuples_ttbar_Dilep/180307_all/Trainings/DeepCSV/generator_174149_full2016/KERAS_check_best_model.h5')
#train.compileModel(learningrate=0.0005,
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])

loss_weight_flavour =K.variable(1.,dtype='float32')
loss_weight_domain = K.variable(0.,dtype='float32')

if not train.modelSet():
    from models import dense_model_gradientReversal

    print 'Setting model'
    train.setModel(dense_model_gradientReversal, dropoutRate=0.1)


    train.compileModel(learningrate=0.0005,
                       loss=['categorical_crossentropy', 'binary_crossentropy'],
                       metrics=['accuracy'],
                       loss_weights = [loss_weight_flavour,loss_weight_domain]
    )

# train without generator
model = train.keras_model
model.summary()

#pdb.set_trace()


from DeepJet_callbacks import DeepJet_callbacks, trainAdversarial

callbacks = DeepJet_callbacks(model,
                              stop_patience=300,
                              lr_factor=0.5,
                              lr_patience=5,
                              lr_epsilon=0.0001,
                              lr_cooldown=2,
                              lr_minimum=0.000001,
                              outputDir=train.outputDir,
                              checkperiod=10)

callback_trainAdversarial = trainAdversarial(model,
                                             x = da_x,
                                             y = [da_y,da_y2],
                                             w = da_w,
                                             start = 0,
                                             step = 10,
                                             batch_size = 50000,
                                             nBatches = 10,
                                             loss_weights = [loss_weight_flavour, loss_weight_domain]
                                             )
cbs = callbacks.callbacks
cbs.append(callback_trainAdversarial)

#pdb.set_trace()

print('starting training')
history = model.fit(x=pretrain_x,
                    y=[pretrain_y,np.zeros(len(pretrain_y))],
                    #                    sample_weight=train_weights,
                    #                    validation_data=(val_features,val_labels,val_weights),
                    validation_split=0.2,
                    epochs=120,
                    callbacks=cbs,
                    batch_size=100000,
                    #sample_weight=[np.ones(len(pretrain_y)),np.zeros(len(pretrain_y))],
                    #                    shuffle=False,
                    #class_weight=class_weight
                    )





pickle.dump(history.history, open(train.outputDir + "Keras_history.p", "wb"))

print("finished training at " + str(datetime.now()))


