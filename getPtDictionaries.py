#!/usr/bin/env python
from __future__ import division
import ROOT
import sys
import os
from ROOT import *
import pickle
import time
from math import  *
import argparse
import collections
import itertools
import string
import glob
import multiprocessing
from array import array
ROOT.gROOT.SetBatch()

parser = argparse.ArgumentParser(description='Get dictionaries for analysing nTuples')
parser.add_argument('-i', "--inDir", dest='inDir', default='/unix/atlasvhbb2/ava/FTAGFramework/athenaOutputs/',
                    help='input directory with merged nTuples from athena')
parser.add_argument('-o', "--outDir", dest='outDir', default='./varDicts/', help='output directory to store pickles')
parser.add_argument('-t', "--tracks", dest='tracks', default='nom:pseudo:ideal', help='input track collections')
parser.add_argument('-v', "--version", dest='version', default='427080_Zprime', help='input sample versions')
args = parser.parse_args()

#nom:nom_replaceHFWithTruth:nom_replaceFRAGWithTruth:nom_replaceFRAGHFWithTruth:nom_replaceFRAGHFGEANTWithTruth:nom_replaceWithTruth:pseudo:ideal
def selectJet(event, jet):
    jet_pt = event.jet_pt[jet]/1000
    jet_eta = abs(event.jet_eta[jet])

    if jet_pt <= 100 or jet_pt >= 5000:
        return False
    if jet_eta >= 2.1:
        return False

    return True

def getJets(pTDict, event, jet):
    if event.jet_LabDr_HadF[jet] == 5:
        pTDict['b'] += 1
    elif event.jet_LabDr_HadF[jet] == 4:
        pTDict['c'] += 1
    else:
        pTDict['l'] += 1

    return pTDict

def getVarValues(pTDict, tree, jet):
    for jetVar in pTDict.keys():
        if 'jet' in jetVar:
            pTDict[jetVar].append(getattr(tree, jetVar)[jet])

    return pTDict

def saveDictionaries(files, outDir, version, track, pTbins):

    allpTJetDict = {}
    for binName, binCuts in pTbins.items(): # initiate empty dictionaries
        allpTJetDict[binName] = variables()

    for f in files:
        file = ROOT.TFile(f)
        tree = file.Get("bTag_AntiKt4EMPFlowJets")
        #tree = file.Get("bTag_AntiKt4EMTopoJets")

        for event in tree: # equivalent to for i in tree.GetEntries() i.e. total events, tree.GetEntry(i) i.e. event
            # getattr(object, 'x') is completely equivalent to object.x.
            for jetVar in allpTJetDict.values()[0].keys():
                if 'jet' in jetVar:
                    getattr(event, jetVar)

            for jet in range(tree.njets):  # loop through jets
                jet_pt = event.jet_pt[jet] / 1000
                for binName, binCuts in pTbins.items():
                    pTDict = allpTJetDict[binName]
                    if len(binCuts) > 1:
                        if jet_pt > binCuts[0]:
                            if jet_pt < binCuts[1]:
                                getJets(pTDict, event, jet)
                                if not selectJet(event, jet):
                                    continue
                                getVarValues(pTDict, tree, jet)
                    if len(binCuts) == 1:
                        if 'le' in binName:
                            if jet_pt <= binCuts[0]:
                                getJets(pTDict, event, jet)
                                if not selectJet(event, jet):
                                    continue
                                getVarValues(pTDict, tree, jet)
                        if 'ge' in binName:
                            if jet_pt >= binCuts[0]:
                                getJets(pTDict, event, jet)
                                if not selectJet(event, jet):
                                    continue
                                getVarValues(pTDict, tree, jet)

    for key, value in allpTJetDict.items():
        outName = outDir + version + '_' + key + '_' + track + '_jetVars.pickle'
        with open(outName, 'wb') as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)



    #print("--- %s seconds ---" % (time.time() - start_time))

def variables():
    jetVars = {
        'jet_LabDr_HadF': [],  # b = 5, c = 4, l = 0
        'jet_jf_llr': [],
        'b': 0,
        'c': 0,
        'l': 0,
        'jet_jf_sig3d': [],
        'jet_jf_nvtx': [],
        'jet_jf_ntrkAtVx': [],
        'jet_jf_nvtx1t': [],

    }

    return jetVars

if __name__ == "__main__":
    if not (os.path.isdir(args.outDir)): os.makedirs(args.outDir)
    processes = []  # multiprocessing

    if '427081' in args.version: sampleType = 'Zprime_Extended'
    if '410470' in args.version: sampleType = 'ttbar'
    if '427080' in args.version: sampleType = 'Zprime'

    tracks = args.tracks.split(':')
    fileDict = {}

    pTRFbins = {
        'le_250': [250],
        '250_400': [250, 400],
        '400_1000': [400, 1000],
        'ge_1000': [1000],
    }

    pTbins = {
        'le_150': [150],
        '150_400': [150, 400],
        '400_1000': [400, 1000],
        '1000_1750': [1000, 1750],
        # 'ge_1750': [1000],
    }

    for track in tracks:
        fileDict[track] = glob.glob(args.inDir + '*' + sampleType + '*/' + track + '/*/flav_Akt4EMPf.root')
        print fileDict[track]

        p = multiprocessing.Process(target=saveDictionaries, args=(fileDict[track], args.outDir, args.version, track,
                                                                   pTRFbins))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()