'''
Created on 07 March 2018

@author: dwalter
'''
import ROOT, numpy

from TrainData import TrainData_Flavour, TrainData_simpleTruth, TrainData_fullTruth, fileTimeOut
from preprocessing import MeanNormZeroPad



class TrainData_deepCSV_raw(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but
        - weights geht filled with the eventweights
        - only jets will be removed when they have no flavor label (e.g. when they are undefined)
        - for real data, all jets will be taken (there is no flavor label)
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)

        self.remove = False
        self.weight = False

        self.eventweightbranch=['event_weight']
        self.registerBranches(self.eventweightbranch)

        self.addBranches(['jet_pt', 'jet_eta',
                          'TagVarCSV_jetNSecondaryVertices',
                          'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR',
                          'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm',
                          'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm',
                          'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks',
                          'TagVarCSV_jetNTracksEtaRel'])

        self.addBranches(['TagVarCSVTrk_trackJetDistVal',
                          'TagVarCSVTrk_trackPtRel',
                          'TagVarCSVTrk_trackDeltaR',
                          'TagVarCSVTrk_trackPtRatio',
                          'TagVarCSVTrk_trackSip3dSig',
                          'TagVarCSVTrk_trackSip2dSig',
                          'TagVarCSVTrk_trackDecayLenVal'],
                         6)

        self.addBranches(['TagVarCSV_trackEtaRel'], 4)

        self.addBranches(['TagVarCSV_vertexMass',
                          'TagVarCSV_vertexNTracks',
                          'TagVarCSV_vertexEnergyRatio',
                          'TagVarCSV_vertexJetDeltaR',
                          'TagVarCSV_flightDistance2dVal',
                          'TagVarCSV_flightDistance2dSig',
                          'TagVarCSV_flightDistance3dVal',
                          'TagVarCSV_flightDistance3dSig'],
                         1)

    def readFromRootFile(self, filename, TupleMeanStd, weighter):
        import ROOT

        fileTimeOut(filename, 120)  # give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples = tree.GetEntries()

        # print('took ', sw.getAndReset(), ' seconds for getting tree entries')


        Tuple = self.readTreeFromRootToTuple(filename)

        x_all = MeanNormZeroPad(filename, TupleMeanStd, self.branches, self.branchcutoffs, self.nsamples)

        weights = Tuple[self.eventweightbranch].astype(float)

        truthtuple = Tuple[self.truthclasses]
        # print(self.truthclasses)
        alltruth = self.reduceTruth(truthtuple)


        ys = alltruth
        flav_sum = ys.sum(axis=1)
        if (flav_sum > 1).any():
            raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)

        if (flav_sum == 0).any():
            print('In file: %s I get %d of %d jets with no flavours assigned!' %(filename, numpy.sum(flav_sum == 0), len(alltruth)))

        mask = (flav_sum == 1)
        if(self.isData):
            print("run on data, take all jets")
            mask.fill(1.)

        print("---TrainData_deepCSV_raw---")
        print(x_all)
        print(alltruth)
        print(weights)

        self.x = [x_all[mask]]
        self.y = [alltruth[mask]]
        self.w = [weights[mask]]


