'''
Created on 20 Feb 2017

@author: jkiesele
'''

from __future__ import print_function


from Weighter import Weighter
from pdb import set_trace
import numpy
import logging

import threading
import multiprocessing

threadingfileandmem_lock=threading.Lock()
#threadingfileandmem_lock.release()
#multiproc_fileandmem_lock=multiprocessing.Lock()

def fileTimeOut(fileName, timeOut):
    '''
    simple wait function in case the file system has a glitch.
    waits until the dir, the file should be stored in/read from, is accessible
    again, or the the timeout
    '''
    import os
    filepath=os.path.dirname(fileName)
    if len(filepath) < 1:
        filepath = '.'
    if os.path.isdir(filepath):
        return
    import time
    counter=0
    print('file I/O problems... waiting for filesystem to become available for '+fileName)
    while not os.path.isdir(filepath):
        if counter > timeOut:
            print('...file could not be opened within '+str(timeOut)+ ' seconds')
        counter+=1
        time.sleep(1)


def _read_arrs_(arrwl,arrxl,arryl,doneVal,fileprefix,tdref=None,randomSeed=None):
    import gc
    gc.collect()

    import h5py
    from sklearn.utils import shuffle
    try:
        idstrs=['w','x','y']
        #print('open file ', fileprefix) #debug
        h5f = h5py.File(fileprefix,'r')
        alllists=[arrwl,arrxl,arryl]
        #print('loop ...') #debug
        for j in range(len(idstrs)):
            fidstr=idstrs[j]
            arl=alllists[j]
            for i in range(len(arl)):
                #print('create filename') #debug
                idstr=fidstr+str(i)
                #print('read direct with id sting ', idstr) #debug
                h5f[idstr].read_direct(arl[i])

                #print('next') #debug
                #shuffle each read-in, but each array with the same seed (keeps right asso)
                if randomSeed:
                    # print('shuffle')
                    arl[i]=shuffle(arl[i], random_state=randomSeed)
                
        doneVal.value=True
        h5f.close()
        del h5f
    except Exception as d:
        print('smth went wrong here ...')
        raise d
    finally:
        if tdref:
            print('remove file from ram disc')
            tdref.removeRamDiskFile()  
    
    
class ShowProgress(object):
    def __init__(self,nsteps,total):
        self.nsteps=nsteps
        self.total=total
        self._stepvec=[]
        for i in range(nsteps):
            self._stepvec.append(float(i+1)*float(total)/float(nsteps))
            
        self._counter=0
        
    def show(self,index):
        if index==0:
            logging.info('0%')
        if index>self._stepvec[self._counter]:
            logging.info(str(int(float(index)/float(self.total)*100))+'%')
            self._counter=self._counter+1
        
    def reset(self):
        self._counter=0
        

class TrainData(object):
    '''
    Base class for batch-wise training of the DNN
    '''
    
    
    
    
    def __init__(self):
        '''
        Constructor
        
        '''
        
        self.treename="deepntuplizer/tree"
        
        self.undefTruth=['isUndefined']#, 'isTau']
        
        self.referenceclass='isB'
        
        self.truthclasses=['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isCC',
                           'isGCC','isUD','isS','isG','isUndefined']

        self.eventweightbranch = ['event_weight']
        self.domainbranch = ['isRealData']

        self.allbranchestoberead=[]
        
        #standard branches
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        self.registerBranches(['jet_pt','jet_eta'])
        self.registerBranches(self.eventweightbranch)
        self.registerBranches(self.domainbranch)


        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'

        self.weight_binX = numpy.array([
                10,25,30,35,40,45,50,60,75,100,
                125,150,175,200,250,300,400,500,
                600,2000],dtype=float)
        
        self.weight_binY = numpy.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
            )
        
        self.reducedtruthclasses=[]
        self.regressiontargetclasses=[]
        
        self.flatbranches=[]
        self.branches=[]
        self.branchcutoffs=[]
                
        self.readthread=None
        self.readdone=None
        
        self.remove=True
        self.weight=False
        
        self.clear()
        
        self.reduceTruth(None)
        
    def __del__(self):
        self.readIn_abort()
        self.clear()
        

    def clear(self):
        self.samplename=''
        self.readIn_abort()
        self.readthread=None
        self.readdone=None
        if hasattr(self, 'x'):
            del self.x
            del self.y
            del self.w
        if hasattr(self, 'w_list'):
            del self.w_list
            del self.x_list
            del self.y_list
            
        self.x=[numpy.array([])]
        self.y=[numpy.array([])]
        self.w=[numpy.array([])]
        
        self.nsamples=None
    
        
    def getInputShapes(self):
        '''
        returns a list for each input shape. In most cases only one entry
        '''
        outl=[]
        for x in self.x:
            outl.append(x.shape)
        shapes=[]
        for s in outl:
            _sl=[]
            for i in range(len(s)):
                if i:
                    _sl.append(s[i])
            s=(_sl)
            if len(s)==0:
                s.append(1)
            shapes.append(s)
        return shapes
        
    def getTruthShapes(self):
        outl=[len(self.getUsedTruth())]
        return outl
    
    
    def getNRegressionTargets(self):
        if not self.regressiontargetclasses:
            return 0
        return len(self.regressiontargetclasses)
    
    def getNClassificationTargets(self):
        return len(self.getUsedTruth())
        
    def addBranches(self, blist, cutoff=1):
        self.branches.append(blist)
        self.registerBranches(blist)
        self.branchcutoffs.append(cutoff)
        
    def registerBranches(self,blist):
        self.allbranchestoberead.extend(blist)
        
    def getUsedTruth(self):
        if len(self.reducedtruthclasses) > 0:
            return self.reducedtruthclasses
        else:
            return self.truthclasses
    
    def addFromRootFile(self,fileName):
        '''
        Adds from a root file and randomly shuffles the input
        '''
        raise Exception('to be implemented')
        #just call read from root (virtual in python??), and mix with existing x,y,weight



    def reduceTruth(self, tuple_in=None):
        self.reducedtruthclasses=self.truthclasses
        if tuple_in is not None:
            return numpy.array(tuple_in.tolist())

    def writeOut(self,fileprefix):
        
        import h5py
        fileTimeOut(fileprefix,120)
        h5f = h5py.File(fileprefix, 'w')
        
        # try "lzf", too, faster, but less compression
        def _writeoutListinfo(arrlist,fidstr,h5F):
            arr=numpy.array([len(arrlist)])
            h5F.create_dataset(fidstr+'_listlength',data=arr)
            for i in range(len(arrlist)):
                idstr=fidstr+str(i)
                h5F.create_dataset(idstr+'_shape',data=arrlist[i].shape)
            
        def _writeoutArrays(arrlist,fidstr,h5F):    
            for i in range(len(arrlist)):
                idstr=fidstr+str(i)
                arr=arrlist[i]
                if "meta" in fileprefix[-4:]:
                    from c_readArrThreaded import writeArray
                    if arr.dtype!='float32':
                        arr=arr.astype('float32')
                    writeArray(arr.ctypes.data,fileprefix[:-4]+fidstr+'.'+str(i),list(arr.shape))
                else:
                    h5F.create_dataset(idstr, data=arr, compression="lzf")
        
        
        arr=numpy.array([self.nsamples],dtype='int')
        h5f.create_dataset('n', data=arr)

        _writeoutListinfo(self.w,'w',h5f)
        _writeoutListinfo(self.x,'x',h5f)
        _writeoutListinfo(self.y,'y',h5f)

        _writeoutArrays(self.w,'w',h5f)
        _writeoutArrays(self.x,'x',h5f)
        _writeoutArrays(self.y,'y',h5f)


        h5f.close()
       
    
        
       
    def __createArr(self,shapeinfo):
        import ctypes
        import multiprocessing
        fulldim=1
        for d in shapeinfo:
            fulldim*=d 
        if fulldim < 0: #catch some weird things that happen when there is a file IO error
            fulldim=0 
        # reserve memory for array
        shared_array_base = multiprocessing.RawArray(ctypes.c_float, int(fulldim))
        shared_array = numpy.ctypeslib.as_array(shared_array_base)#.get_obj())
        #print('giving shape',shapeinfo)
        shared_array = shared_array.reshape(shapeinfo)
        #print('gave shape',shapeinfo)
        return shared_array
    
    def removeRamDiskFile(self):
        if hasattr(self, 'ramdiskfile'):
            import os
            try:
                if self.ramdiskfile and os.path.exists(self.ramdiskfile):
                    if "meta" in self.ramdiskfile[-4:]:
                        os.system('rm -f '+self.ramdiskfile[:-4]+"*")
                    else:
                        os.remove(self.ramdiskfile)
            except OSError:
                pass
            self.ramdiskfile=None
               
    def readIn_async(self,fileprefix,read_async=True,shapesOnly=False,ramdiskpath='',randomseed=None):
        print('readIn_async ', fileprefix)
        if self.readthread and read_async:
            print('\nTrainData::readIn_async: started new read before old was finished. Intended? Waiting for first to finish...\n')
            self.readIn_join()
            
        #print('read')
        
        import h5py
        import multiprocessing
        
        #print('\ninit async read\n')
        
        fileTimeOut(fileprefix,120)
        #print('\nfile access ok\n')
        self.samplename=fileprefix
        
        
        
        def _readListInfo_(idstr):
            sharedlist=[]
            shapeinfos=[]
            wlistlength=self.h5f[idstr+'_listlength'][0]
            #print(idstr,'list length',wlistlength)
            for i in range(wlistlength):
                sharedlist.append(numpy.array([]))
                iidstr=idstr+str(i)
                shapeinfo=numpy.array(self.h5f[iidstr+'_shape'])
                shapeinfos.append(shapeinfo)
            return sharedlist, shapeinfos
        
        
        with threadingfileandmem_lock:
            try:
                self.h5f = h5py.File(fileprefix,'r')
            except:
                raise IOError('File %s could not be opened properly, it may be corrupted' % fileprefix)
            self.nsamples=self.h5f['n']
            self.nsamples=self.nsamples[0]
            if True or not hasattr(self, 'w_shapes'):
                self.w_list,self.w_shapes=_readListInfo_('w')
                self.x_list,self.x_shapes=_readListInfo_('x')
                self.y_list,self.y_shapes=_readListInfo_('y')

            else:
                print('\nshape known\n')
                self.w_list,_=_readListInfo_('w')
                self.x_list,_=_readListInfo_('x')
                self.y_list,_=_readListInfo_('y')

                
            self.h5f.close()
            del self.h5f
            self.h5f=None
            if shapesOnly:
                return
            
            readfile=fileprefix
            
            isRamDisk=len(ramdiskpath)>0
            if isRamDisk:
                import shutil
                import uuid
                import os
                import copy
                unique_filename=''
                
                unique_filename = ramdiskpath+'/'+str(uuid.uuid4())+'.z'
                if "meta" in readfile[-4:]:
                    filebase=readfile[:-4]
                    unique_filename = ramdiskpath+'/'+str(uuid.uuid4())
                    shutil.copyfile(filebase+'meta',unique_filename+'.meta')
                    for i in range(len(self.w_list)):
                        shutil.copyfile(filebase+'w.'+str(i),unique_filename+'.w.'+str(i))
                    for i in range(len(self.x_list)):
                        shutil.copyfile(filebase+'x.'+str(i),unique_filename+'.x.'+str(i))
                    for i in range(len(self.y_list)):
                        shutil.copyfile(filebase+'y.'+str(i),unique_filename+'.y.'+str(i))
                    unique_filename+='.meta'
                        
                else:
                    unique_filename = ramdiskpath+'/'+str(uuid.uuid4())+'.z'
                    shutil.copyfile(fileprefix, unique_filename)
                readfile=unique_filename
                self.ramdiskfile=readfile

            #create shared mem in sync mode
            for i in range(len(self.w_list)):
                self.w_list[i]=self.__createArr(self.w_shapes[i])
                
            for i in range(len(self.x_list)):
                self.x_list[i]=self.__createArr(self.x_shapes[i])
                
            for i in range(len(self.y_list)):
                self.y_list[i]=self.__createArr(self.y_shapes[i])
            
            if read_async:
                self.readdone=multiprocessing.Value('b',False)
                        
        if read_async:
            if "meta" in readfile[-4:]:
                #new format
                from c_readArrThreaded import startReading
                self.readthreadids=[]
                filebase=readfile[:-4]
                for i in range(len(self.w_list)):
                    self.readthreadids.append(startReading(self.w_list[i].ctypes.data,
                                                           filebase+'w.'+str(i),
                                                           list(self.w_list[i].shape),
                                                           isRamDisk))
                for i in range(len(self.x_list)):
                    self.readthreadids.append(startReading(self.x_list[i].ctypes.data,
                                                           filebase+'x.'+str(i),
                                                           list(self.x_list[i].shape),
                                                           isRamDisk))
                for i in range(len(self.y_list)):
                    self.readthreadids.append(startReading(self.y_list[i].ctypes.data,
                                                           filebase+'y.'+str(i),
                                                           list(self.y_list[i].shape),
                                                           isRamDisk))
                
                
            else:
                self.readthread=multiprocessing.Process(target=_read_arrs_, 
                                                        args=(self.w_list,
                                                              self.x_list,
                                                              self.y_list,
                                                              self.readdone,
                                                              readfile,
                                                              self,randomseed))
                self.readthread.start()
        else:
            if "meta" in readfile[-4:]:
                from c_readArrThreaded import readBlocking
                filebase=readfile[:-4]
                self.readthreadids=[]
                for i in range(len(self.w_list)):
                    (readBlocking(self.w_list[i].ctypes.data,
                                                           filebase+'w.'+str(i),
                                                           list(self.w_list[i].shape),
                                                           isRamDisk))
                for i in range(len(self.x_list)):
                    (readBlocking(self.x_list[i].ctypes.data,
                                                           filebase+'x.'+str(i),
                                                           list(self.x_list[i].shape),
                                                           isRamDisk))
                for i in range(len(self.y_list)):
                    (readBlocking(self.y_list[i].ctypes.data,
                                                           filebase+'y.'+str(i),
                                                           list(self.y_list[i].shape),
                                                           isRamDisk))
                
            else:
                self.readdone=multiprocessing.Value('b',False)
                _read_arrs_(self.w_list,self.x_list,self.y_list,self.readdone,readfile,self,randomseed)

            
            
        
    def readIn_abort(self):
        self.removeRamDiskFile()
        if not self.readthread:
            return
        self.readthread.terminate()
        self.readthread=None
        self.readdone=None
     
    def readIn_join(self,wasasync=True,waitforStart=True):
        
        try:
            if not not hasattr(self, 'readthreadids') and not waitforStart and not self.readthread and wasasync:
                print('\nreadIn_join:read never started\n')
            
            import time
            if waitforStart:
                while (not hasattr(self, 'readthreadids')) and not self.readthread:
                    time.sleep(0.1)
                if hasattr(self, 'readthreadids'):
                    while not self.readthreadids:
                        time.sleep(0.1)
            
            counter=0
            
            if hasattr(self, 'readthreadids') and self.readthreadids:
                from c_readArrThreaded import isDone
                doneids=[]
                while True:
                    for id in self.readthreadids:
                        if id in doneids: continue
                        if isDone(id):
                            doneids.append(id)
                    if len(self.readthreadids) == len(doneids):
                        break
                    time.sleep(0.1)
                    counter+=1
                    if counter>3000: #read failed. do synchronous read, safety option if threads died
                        print('\nfalling back to sync read\n')
                        self.readthread.terminate()
                        self.readthread=None
                        self.readIn(self.samplename)
                        return
                
            else: #will be removed at some point
                while wasasync and (not self.readdone or not self.readdone.value): 
                    if not self.readthread:
                        time.sleep(.1)
                        continue
                    self.readthread.join(.1)
                    counter+=1
                    if counter>3000: #read failed. do synchronous read, safety option if threads died
                        print('\nfalling back to sync read\n')
                        self.readthread.terminate()
                        self.readthread=None
                        self.readIn(self.samplename)
                        return
                if self.readdone.value:
                    self.readthread.join(.1)
                    
            import copy
            #move away from shared memory
            #this costs performance but seems necessary
            direct=False
            with threadingfileandmem_lock:
                if direct:
                    self.w=self.w_list
                    self.x=self.x_list
                    self.y=self.y_list
                else:
                    self.w=copy.deepcopy(self.w_list)
                    self.x=copy.deepcopy(self.x_list)
                    self.y=copy.deepcopy(self.y_list)
                    
                del self.w_list
                del self.x_list
                del self.y_list
            #in case of some errors during read-in
            
        except Exception as d:
            raise d
        finally:
            self.removeRamDiskFile()
        
        #check if this is really neccessary 
        def reshape_fast(arr,shapeinfo):
            if len(shapeinfo)<2:
                shapeinfo=numpy.array([arr.shape[0],1])
            arr=arr.reshape(shapeinfo)
            return arr
        
        
        for i in range(len(self.w)):
            self.w[i]=reshape_fast(self.w[i],self.w_shapes[i])
        for i in range(len(self.x)):
            self.x[i]=reshape_fast(self.x[i],self.x_shapes[i])
        for i in range(len(self.y)):
            self.y[i]=reshape_fast(self.y[i],self.y_shapes[i])
        
        self.w_list=None
        self.x_list=None
        self.y_list=None
        if wasasync and self.readthread:
            self.readthread.terminate()
        self.readthread=None
        self.readdone=None
        
    def readIn(self,fileprefix,shapesOnly=False):
        self.readIn_async(fileprefix,False,shapesOnly)
        direct=True
        if direct:
            self.w=self.w_list
            self.x=self.x_list
            self.y=self.y_list
        else:
            import copy
            self.w=copy.deepcopy(self.w_list)
            del self.w_list
            self.x=copy.deepcopy(self.x_list)
            del self.x_list
            self.y=copy.deepcopy(self.y_list)
            del self.y_list
        
        def reshape_fast(arr,shapeinfo):
            if len(shapeinfo)<2:
                shapeinfo=numpy.array([arr.shape[0],1])
            if shapesOnly:
                arr=numpy.zeros(shape=shapeinfo)
            else:
                arr=arr.reshape(shapeinfo)
            return arr
        
        
            
            
        for i in range(len(self.w)):
            self.w[i]=reshape_fast(self.w[i],self.w_shapes[i])
        for i in range(len(self.x)):
            self.x[i]=reshape_fast(self.x[i],self.x_shapes[i])
        for i in range(len(self.y)):
            self.y[i]=reshape_fast(self.y[i],self.y_shapes[i])
        
        self.w_list=None
        self.x_list=None
        self.y_list=None
        self.readthread=None
        
        
    def readTreeFromRootToTuple(self, filenames, limit=None, branches=None):
        '''
        To be used to get the initial tupel for further processing in inherting classes
        Makes sure the number of entries is properly set
        
        can also read a list of files (e.g. to produce weights/removes from larger statistics
        (not fully tested, yet)
        '''
        if  branches==None:
            branches=self.allbranchestoberead
            
        #print(branches)
        #remove duplicates
        usebranches=list(set(branches))
        tmpbb=[]
        for b in usebranches:
            if len(b):
                tmpbb.append(b)
        usebranches=tmpbb
            
        import ROOT
        from root_numpy import tree2array, root2array
        if isinstance(filenames, list):
            for f in filenames:
                fileTimeOut(f,120)
            print('add files')
            nparray = root2array(
                filenames, 
                treename = self.treename, 
                stop = limit,
                branches = usebranches
                )
            print('done add files')
            return nparray
            print('add files')
        else:    
            fileTimeOut(filenames,120) #give eos a minute to recover
            rfile = ROOT.TFile(filenames)
            tree = rfile.Get(self.treename)
            if not self.nsamples:
                self.nsamples=tree.GetEntries()
            nparray = tree2array(tree, stop=limit, branches=usebranches)
            return nparray
        
    def make_means(self, nparray):
        from preprocessing import meanNormProd
        return meanNormProd(nparray)
        
    def produceMeansFromRootFile(self,filename, limit=500000):
        from preprocessing import meanNormProd
        nparray = self.readTreeFromRootToTuple(filename, limit=limit)
        means = self.make_means(nparray)
        del nparray
        return means
    
    #overload if necessary
    def make_empty_weighter(self):
        from Weighter import Weighter
        weighter = Weighter() 
        weighter.undefTruth = self.undefTruth
        
        if self.remove or self.weight:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truthclasses
                )
        return weighter

       
    def produceBinWeighter(self,filenames):
        weighter = self.make_empty_weighter()
        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truthclasses)
        showprog=ShowProgress(5,len(filenames))
        counter=0
        if self.remove or self.weight:
            for fname in filenames:
                nparray = self.readTreeFromRootToTuple(fname, branches=branches)
                weighter.addDistributions(nparray)
                del nparray
                showprog.show(counter)
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
        return weighter
    
        
    def getFlavourClassificationData(self,filename,TupleMeanStd, weighter):
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        #print('took ', sw.getAndReset(), ' seconds for getting tree entries')
    
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        x_all = MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        #print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        notremoves=numpy.array([])
        weights=numpy.array([])
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            weights=notremoves
            #print('took ', sw.getAndReset(), ' to create remove indices')
        elif self.weight:
            #print('creating weights')
            weights= weighter.getJetWeights(Tuple)
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            #print('remove')
            weights=weights[notremoves > 0]
            x_all=x_all[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_all.shape[0]
        #print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        #print('took in total ', swall.getAndReset(),' seconds for conversion')
        
        return weights,x_all,alltruth, notremoves

    def _normalize_input_(self, weighter, npy_array):
        weights = None
        if self.weight:
            weights=weighter.getJetWeights(npy_array)
            self.w = [weights for _ in self.y]
        elif self.remove:
            notremoves=weighter.createNotRemoveIndices(npy_array)
            undef=npy_array['isUndefined']
            notremoves-=undef
            print(' to created remove indices')
            weights=notremoves

            print('remove')
            self.x = [x[notremoves > 0] for x in self.x]
            self.y = [y[notremoves > 0] for y in self.y]
            weights=weights[notremoves > 0]
            self.w = [weights for _ in self.y]
            newnsamp=self.x[0].shape[0]
            print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
            self.nsamples = newnsamp
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
            self.w = [weights for _ in self.y]        

        

from preprocessing import MeanNormApply, MeanNormZeroPad

class TrainData_Flavour(TrainData):
    '''
    
    '''
    def __init__(self):
        TrainData.__init__(self)
        self.clear()
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        weights,x_all,alltruth, _ =self.getFlavourClassificationData(filename,TupleMeanStd, weighter)
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[alltruth]
        
     
     
class TrainData_simpleTruth(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isC','isUDSG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            bb = tuple_in['isBB'].view(numpy.ndarray)
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            allb = b+bl+blc
            
           
            c = tuple_in['isC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            g = tuple_in['isG'].view(numpy.ndarray)
            l = g + uds
            return numpy.vstack((allb,bb,c,l)).transpose()
    
    
class TrainData_leptTruth(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDSG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            bb = tuple_in['isBB'].view(numpy.ndarray)
            gbb = tuple_in['isGBB'].view(numpy.ndarray)
            
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            lepb=bl+blc
           
            c = tuple_in['isC'].view(numpy.ndarray)
            cc = tuple_in['isCC'].view(numpy.ndarray)
            gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            l = g + uds
            
            return numpy.vstack((b,bb+gbb,lepb,c+cc+gcc,l)).transpose()  
        
        
        

class TrainData_fullTruth(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDS','isG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            
            bb = tuple_in['isBB'].view(numpy.ndarray)
            gbb = tuple_in['isGBB'].view(numpy.ndarray)
            
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            lepb=bl+blc
           
            c = tuple_in['isC'].view(numpy.ndarray)
            cc = tuple_in['isCC'].view(numpy.ndarray)
            gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            
            
            return numpy.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()    
  

class TrainData_QGOnly(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.clear()
        self.undefTruth=['isUndefined']
        
        self.referenceclass='isUD'
        
        
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isUDS','isG']
        if tuple_in is not None:
            #b = tuple_in['isB'].view(numpy.ndarray)
            #bb = tuple_in['isBB'].view(numpy.ndarray)
            #gbb = tuple_in['isGBB'].view(numpy.ndarray)
            #
            #
            #bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            #blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            #lepb=bl+blc
            #
            #c = tuple_in['isC'].view(numpy.ndarray)
            #cc = tuple_in['isCC'].view(numpy.ndarray)
            #gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            
            
            return numpy.vstack((uds,g)).transpose()    

class TrainData_quarkGluon(TrainData):
    def __init__(self):
        super(TrainData_quarkGluon, self).__init__()
        self.referenceclass = 'isG'
        self.reducedtruthclasses=['isQ', 'isG']
        self.clear()
        
    def reduceTruth(self, tuple_in):
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            #bb = tuple_in['isBB'].view(numpy.ndarray) #this should be gluon?
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            c = tuple_in['isC'].view(numpy.ndarray)
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            q = ud+s#+c+blc+bl+b
            
            g = tuple_in['isG'].view(numpy.ndarray)
            return numpy.vstack((q, g)).transpose()    
        else:
            print('I got an empty tuple?')
