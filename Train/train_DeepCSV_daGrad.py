
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


os.environ["CUDA_VISIBLE_DEVICES"] = '3'
gpu_options = tf.GPUOptions(allow_growth=True)  # ,per_process_gpu_memory_fraction=0.1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

from Helpers import shuffle_in_unison, get_class_weights

print( "start training at " +str(datetime.now()))

### Gather the samples together

print('load sample files ...')
f = h5py.File(
    '/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/Pretraining_Thu_174149_full2016/HDF5/ntuples_unshuffled.hdf5',
    #'/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_TT/HDF5/ntuples.hdf5',
    'r')
pretrain_y = f['y'][()]
pretrain_x = f['x'][()]
f.close()

f = h5py.File(
    '/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_Data/HDF5/ntuples.hdf5',
    'r')
data_y = f['y'][()]
data_x = f['x'][()]
data_w = f['w'][()]
data_y2 = np.ones(len(data_y))
f.close()

mc_y = []
mc_x = []
mc_w = []

def fill_mc(filename, weight = 1.):
    f = h5py.File(
    filename,
    'r')
    mc_y.append(f['y'][()])
    mc_x.append(f['x'][()])
    mc_w.append(f['w'][()]*weight)
    f.close()



fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_TT/HDF5/ntuples.hdf5', 831760*1./(77081156 + 77867738))
fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_DY50/HDF5/ntuples.hdf5', 6025200*1./81781052)
fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_DY10to50/HDF5/ntuples.hdf5', 22635100*1./47946519)
fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_WW/HDF5/ntuples.hdf5', 118700*1./994012)
fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_WZ/HDF5/ntuples.hdf5', 44900*1./1000000)
fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_ZZ/HDF5/ntuples.hdf5', 15400*1./998034)
fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_Wantit/HDF5/ntuples.hdf5', 35600*1./6933094)
fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_Wt/HDF5/ntuples.hdf5', 35600*1./6952830)
fill_mc('/local/scratch/ssd1/dwalter/data/domainAdaptionStudies/180307_all/Traindata/ttbarSelected/DC_WJets/HDF5/ntuples.hdf5', 61526700*1./16497031)


mc_y = np.vstack(mc_y)
mc_x = np.vstack(mc_x)
mc_w = np.vstack(mc_w)
mc_y2 = np.zeros(len(mc_y))

norm = np.sum(data_w)/np.sum(mc_w)
print("norm = " +str(norm))
mc_w = mc_w * norm

da_x = np.vstack((data_x,mc_x))
da_y = np.vstack((data_y,mc_y))
da_y2 = np.append(data_y2,mc_y2)
da_w = np.vstack((data_w,mc_w))

da_x, da_y, da_y2, da_w = shuffle_in_unison((da_x,da_y,da_y2,da_w))


print("got all the samples!")

#pdb.set_trace()

### Training part
# also does all the parsing
train= training_base(testrun=False)
print 'Inited'




# print('shuffle samples in union')
# pretrain_y, pretrain_x, pretrain_w = shuffle_in_unison([pretrain_y, pretrain_x, pretrain_w])

#train.loadModel('/local/scratch/ssd1/dwalter/data/Ntuples_ttbarSelected/180307_all/Trainings/DeepCSV/generator_174149_full2016/KERAS_check_best_model.h5')
#train.compileModel(learningrate=0.0005,
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])

loss_weight_flavour =K.variable(1.,dtype='float32')
loss_weight_domain = K.variable(0.,dtype='float32')

if not train.modelSet():
    from models import dense_model_gradientReversal_seq

    print 'Setting model'
    train.setModel(dense_model_gradientReversal_seq, dropoutRate=0.1)


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
                              lr_cooldown=5,
                              lr_minimum=0.000001,
                              outputDir=train.outputDir,
                              checkperiod=10)

callback_trainAdversarial = trainAdversarial(model,
                                             x = da_x,
                                             y = [da_y,da_y2],
                                             w = da_w,
                                             start = 10,
                                             step = 5,
                                             batch_size = 50000,
                                             loss_weights = [loss_weight_flavour, loss_weight_domain]
                                             )
cbs = callbacks.callbacks
cbs.append(callback_trainAdversarial)


print('starting training')
history = model.fit(x=pretrain_x,
                    y=[pretrain_y,np.zeros(len(pretrain_y))],
                    #                    sample_weight=train_weights,
                    #                    validation_data=(val_features,val_labels,val_weights),
                    validation_split=0.2,
                    epochs=200,
                    callbacks=cbs,
                    batch_size=500000,
                    #sample_weight=[np.ones(len(pretrain_y)),np.zeros(len(pretrain_y))],
                    #                    shuffle=False,
                    #                    class_weight=get_class_weights(pretrain_y)
                    )



pickle.dump(history.history, open(train.outputDir + "Keras_history.p", "wb"))

print("finished training at " + str(datetime.now()))


