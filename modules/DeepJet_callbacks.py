'''
Created on 7 Apr 2017

@author: jkiesele
'''
from __future__ import print_function

from ReduceLROnPlateau import ReduceLROnPlateau
from keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint #, ReduceLROnPlateau # , TensorBoard
# loss per epoch
from time import time
from pdb import set_trace
import json
from keras import backend as K

class newline_callbacks_begin(Callback):
    
    def __init__(self,outputDir,plotLoss=False):
        self.outputDir=outputDir
        self.loss=[]
        self.val_loss=[]
        self.full_logs=[]
        self.plotLoss=plotLoss
        
    def on_epoch_end(self,epoch, epoch_logs={}):
        import os
        lossfile=os.path.join( self.outputDir, 'losses.log')
        print('\n***callbacks***\nsaving losses to '+lossfile)
        self.loss.append(epoch_logs.get('loss'))
        self.val_loss.append(epoch_logs.get('val_loss'))
        f = open(lossfile, 'a')
        f.write(str(epoch_logs.get('loss')))
        f.write(" ")
        f.write(str(epoch_logs.get('val_loss')))
        f.write("\n")
        f.close()    
        learnfile=os.path.join( self.outputDir, 'learn.log')
        with open(learnfile, 'a') as f:
            f.write(str(float(K.get_value(self.model.optimizer.lr)))+'\n')
        
        normed = {}
        for vv in epoch_logs:
            normed[vv] = float(epoch_logs[vv])
        self.full_logs.append(normed)
        lossfile=os.path.join( self.outputDir, 'full_info.log')
        with open(lossfile, 'w') as out:
            out.write(json.dumps(self.full_logs))
            
        if self.plotLoss:
            from testing import plotLossAndLearn
            plotLossAndLearn(self.outputDir+'/losses.log',self.outputDir+'/learn.log',self.outputDir+'/losses.pdf',[])
        
class newline_callbacks_end(Callback):
    def on_epoch_end(self,epoch, epoch_logs={}):
        print('\n***callbacks end***\n')

class newline_callbacks_batchloss(Callback):
    def on_batch_end(self,batch,batch_logs={}):
        print('\n loss ... '+str(batch_logs.get('loss')))
        
        
class Losstimer(Callback):
    def __init__(self, every = 5):
        self.points = []
        self.every = every

    def on_train_begin(self, logs):
        self.start = time()

    def on_batch_end(self, batch, logs):
        if (batch % self.every) != 0: return
        elapsed = time() - self.start
        cop = {}
        for i, j in logs.iteritems():
            cop[i] = float(j)
        cop['elapsed'] = elapsed
        self.points.append(cop)
        
        
class checkTokens_callback(Callback):
    
    def __init__(self,cutofftime_hours=48):
        self.cutofftime_hours=cutofftime_hours
        
    def on_epoch_begin(self, epoch, logs=None):
        from tokenTools import checkTokens
        checkTokens(self.cutofftime_hours)
        
class saveCheckPointDeepJet(Callback):
    '''
    this seems obvious, however for some reason the keras model checkpoint fails
    to save the optimizer state, needed for resuming a training. Therefore this explicit
    implementation.
    '''
    
    def __init__(self,outputDir,model):
        self.outputDir=outputDir
        self.djmodel=model
    def on_epoch_end(self,epoch, epoch_logs={}):
        self.djmodel.save(self.outputDir+"/KERAS_check_model_last.h5")


class outputDistribution(Callback):
    def __init__(self, generator, steps, outputDir):
        self.generator = generator
        self.steps = steps
        self.outputDir=outputDir



    def on_train_end(self, epoch, epoch_logs={}):

        y_pred = self.model.predict_generator(self.generator, steps=self.steps)
        from testing import plotOutput
        plotOutput(y_pred,self.outputDir)

class trainAdversarial(Callback):
    def __init__(self, model, x, y, w, start=0, step = 2, batch_size=500000, epochs = 1,loss_weights = [1.,1.]):
        self.model = model
        self.x = x
        self.y = y
        self.w = w

        self.start = start
        self.step = step
        self.batch_size = batch_size
        self.epochs = epochs

        self.loss_weight_flavour = loss_weights[0]
        self.loss_weight_domain = loss_weights[1]

        self.epoch_number = 0

    def on_epoch_begin(self,epoch, epoch_logs={}):
        self.epoch_number = epoch

    def on_batch_end(self,batch, batch_logs={}):
        import numpy as np
        if self.epoch_number >= self.start and batch%self.step == 0:
            print("train adversarial")


            K.set_value(self.loss_weight_flavour, 0.)
            K.set_value(self.loss_weight_domain, 1.)

            self.model.fit(x=self.x,
                           y=self.y,
                           sample_weight =[np.ones(len(self.w)),np.squeeze(self.w)],
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           )

            K.set_value(self.loss_weight_flavour, 1.)
            K.set_value(self.loss_weight_domain, 0.)

        
class DeepJet_callbacks(object):
    def __init__(self,
                 model,
                 stop_patience=10,
                 lr_factor=0.5,
                 lr_patience=1,
                 lr_epsilon=0.001,
                 lr_cooldown=4,
                 lr_minimum=1e-5,
                 outputDir='',
                 minTokenLifetime=5,
                 checkperiod=10,
                 plotLossEachEpoch=True):


        #self.pred=outputDistribution(generator,generatorsteps,outputDir)

        self.nl_batchloss=newline_callbacks_batchloss()

        self.nl_begin=newline_callbacks_begin(outputDir,plotLossEachEpoch)
        self.nl_end=newline_callbacks_end()
        
        self.stopping = EarlyStopping(monitor='val_loss', 
                                      patience=stop_patience, 
                                      verbose=1, mode='min')
        
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, 
                                mode='min', verbose=1, epsilon=lr_epsilon,
                                 cooldown=lr_cooldown, min_lr=lr_minimum)

        self.modelbestcheck=ModelCheckpoint(outputDir+"/KERAS_check_best_model.h5", 
                                        monitor='val_loss', verbose=1, 
                                        save_best_only=True, save_weights_only=True)
        
        self.modelcheckperiod=ModelCheckpoint(outputDir+"/KERAS_check_model_epoch{epoch:02d}.h5", verbose=1,period=checkperiod, save_weights_only=True)
        
        self.modelcheck=saveCheckPointDeepJet(outputDir,model)
        
        
  
        self.history=History()
        self.timer = Losstimer()
        
        self.tokencheck=checkTokens_callback(minTokenLifetime)
  
        self.callbacks=[
            self.nl_begin, self.tokencheck,
            self.modelbestcheck,
            #self.modelcheck,
            self.modelcheckperiod,
            self.reduce_lr, self.stopping, self.nl_end, self.history,
            self.timer,
            #self.nl_batchloss
        ]
