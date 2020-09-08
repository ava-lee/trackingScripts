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
parser.add_argument('-d', "--outDir", dest='outDir', default='./varDicts/', help='output directory to store pickles')
args = parser.parse_args()


def selectJet(event, jet):
    jet_pt = event.jet_pt[jet]/1000
    jet_eta = abs(event.jet_eta[jet])

    if jet_pt < 100 or jet_pt > 5000:
        return False
    if jet_eta < 0 or jet_eta > 2.1:
        return False

    return True

def saveDictionaries(filepath, outDir, version, track, jetVars="", trackVars="", comment=""):
    file = ROOT.TFile(str(filepath))
    tree = file.Get("bTag_AntiKt4EMPFlowJets") #tree = file.Get("bTag_AntiKt4EMTopoJets")

    for event in tree: # equivalent to for i in tree.GetEntries() i.e. total events, tree.GetEntry(i) i.e. event
        # getattr(object, 'x') is completely equivalent to object.x.
        if jetVars !="":
            for jetVar in jetVars.keys():
                getattr(event, jetVar)
        if trackVars !="":
            for trackVar in trackVars.keys():
                getattr(event, trackVar)

        for jet in range(tree.njets): # loop through jets
            if not selectJet(event, jet): # if not baseline and jet selection
                continue

            if jetVars != "":
                for jetVar in jetVars.keys():
                    if jetVar == "jet_jf_m": jetVars[jetVar].append(getattr(tree,jetVar)[jet]/1000)
                    else: jetVars[jetVar].append(getattr(tree,jetVar)[jet])

            if trackVars != "":
                for btrack in range(tree.jet_btag_ntrk[jet]): # loop through b-tagged tracks within jets to get track info
                    if event.jet_trk_orig[jet][btrack] == 0 or event.jet_trk_orig[jet][btrack] == 1: # track selection
                        for trackVar in trackVars.keys():
                            if trackVar == 'jet_LabDr_HadF': trackVars[trackVar].append(getattr(tree, trackVar)[jet])
                            elif trackVar == 'jet_dRiso': trackVars[trackVar].append(getattr(tree, trackVar)[jet])
                            else: trackVars[trackVar].append(getattr(tree, trackVar)[jet][btrack])
    if jetVars != "":
        jetVarName = "jetVars"
        if comment != "": jetVarName += '_' + comment
        outJetVarsName = outDir + version + '_' + track + '_' + jetVarName + '.pickle'
        with open(outJetVarsName, 'wb') as jetHandle:
            pickle.dump(jetVars, jetHandle, protocol=pickle.HIGHEST_PROTOCOL)

    if trackVars !="":
        trackVarName = "trackVars"
        if comment != "": trackVarName += '_' + comment
        outTrackVarsName = outDir + version + '_' + track + '_' + trackVarName + '.pickle'
        with open(outTrackVarsName, 'wb') as trackHandle:
            pickle.dump(trackVars, trackHandle, protocol=pickle.HIGHEST_PROTOCOL)

    #print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    if not (os.path.isdir(args.outDir)): os.makedirs(args.outDir)
    files = glob.glob(args.inDir+'*/*/all_flav*.root')
    processes = [] #multiprocessing

    for i in range(len(files)):
        track = files[i].split('_')[-1].replace('.root','')
        version = files[i].split('.')[2] + '_' + files[i].split('.')[3].split('_')[-1]

        jetVars = {
            'jet_LabDr_HadF': [],  # to categorise jets
            'jet_jf_n2t': [],
            'jet_jf_m': [],
            'jet_jf_sig3d': [],
            'jet_jf_efc': [],
            'jet_jf_nvtx1t': [],
            'jet_jf_nvtx': [],
            'jet_jf_ntrkAtVx': [],
            'jet_jf_dR': [],
            'jet_dRiso': [], # to categorise jets
        }

        trackVars = {
            'jet_trk_truthMatchProbability': [],
            'jet_trk_nPixHits': [],
            'jet_trk_nSCTHits': [],
            'jet_trk_nsplitPixHits': [],
            'jet_trk_nsharedPixHits': [],
            'jet_trk_nsharedSCTHits': [],
            'jet_trk_jf_Vertex': [],
            'jet_LabDr_HadF': [], # to categorise tracks
            'jet_dRiso': [], # to categorise tracks
        }

        p = multiprocessing.Process(target=saveDictionaries, args=(files[i], args.outDir, version, track, jetVars, "", ))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()