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
parser.add_argument("--jetVars", action='store_true', dest='jetVars', default=False,
                    help="Write jet information to output file")
parser.add_argument("--trackVars", action='store_true', dest='trackVars', default=False,
                    help="Write track information to output file")
parser.add_argument("--pTsplit", action='store_true', dest='pTsplit', default=False,
                    help="If should separate into different pT bins")
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

def selectTrack(event, jet, track):
    pt = event.jet_trk_pt[jet][track]/1000
    d0 = abs(event.jet_trk_d0[jet][track])
    z0theta = event.jet_trk_z0[jet][track] * sin(event.jet_trk_theta[jet][track])

    if pt <= 0.5:
        return False
    if d0 >= 7.0:
        return False
    if z0theta >= 10:
        return False
    if event.jet_trk_nPixHits[jet][track] < 1:
        return False
    if event.jet_trk_nSCTHits[jet][track] < 4:
        return False
    if (event.jet_trk_nPixHits[jet][track] + event.jet_trk_nSCTHits[jet][track]) < 7:
        return False
    if (event.jet_trk_nsharedPixHits[jet][track] + floor(event.jet_trk_nsharedSCTHits[jet][track]/2)) > 1:
        return False
    if event.jet_trk_nPixHoles[jet][track] > 1:
        return False
    if (event.jet_trk_nPixHoles[jet][track] + event.jet_trk_nSCTHoles[jet][track]) > 2:
        return False

    return True


def saveDictionaries(files, outDir, version, track, jetVars="", trackVars="", vtxVars="", comment=""):
    count = 0
    b = 0
    c = 0
    l = 0

    for f in files:
        file = ROOT.TFile(f)
        tree = file.Get("bTag_AntiKt4EMPFlowJets")
        #tree = file.Get("bTag_AntiKt4EMTopoJets")

        for event in tree: # equivalent to for i in tree.GetEntries() i.e. total events, tree.GetEntry(i) i.e. event
            # getattr(object, 'x') is completely equivalent to object.x.
            count += 1
            if jetVars !="":
                for jetVar in jetVars.keys():
                    getattr(event, jetVar)
            if vtxVars !="":
                for vtxVar in vtxVars.keys():
                    getattr(event, vtxVar)
            if trackVars !="":
                for trackVar in trackVars.keys():
                    if trackVar == 'jetEventNumber':
                        continue
                    getattr(event, trackVar)
            #print count

            for jet in range(tree.njets): # loop through jets
                if event.jet_LabDr_HadF[jet] == 5:
                    b += 1
                elif event.jet_LabDr_HadF[jet] == 4:
                    c += 1
                else:
                    l += 1

                if not selectJet(event, jet):
                    continue

                if jetVars != "":
                    for jetVar in jetVars.keys():
                        if jetVar == "jet_pt": jetVars[jetVar].append(getattr(tree,jetVar)[jet]/1000)
                        else: jetVars[jetVar].append(getattr(tree,jetVar)[jet])

                #if vtxVars != "":
                    #for vtxVar in vtxVars.keys():
                        #print jet, vtxVar, getattr(tree,vtxVar)[jet]

                if trackVars != "":
                    index = float(str(count) + "." + str(jet))  # for jet-track association
                    for btrack in range(tree.jet_btag_ntrk[jet]): # loop through b-tagged tracks within jets to get track info
                        if not selectTrack(event, jet, btrack):
                            continue
                        #print tree.jet_btag_ntrk[jet]
                        for trackVar in trackVars.keys():
                            if trackVar == 'jetEventNumber': trackVars[trackVar].append(index)
                            elif 'trk' in trackVar: trackVars[trackVar].append(getattr(tree, trackVar)[jet][btrack])
                            else: trackVars[trackVar].append(getattr(tree, trackVar)[jet])
            #print jetVars

    if jetVars != "":
        jetVars['b'] = b
        jetVars['c'] = c
        jetVars['l'] = l
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
    processes = []  # multiprocessing

    if '427081' in args.version: sampleType = 'Zprime_Extended'
    if '410470' in args.version: sampleType = 'ttbar'
    if '427080' in args.version: sampleType = 'Zprime'

    # If specific folder, need to specify the sample type folder too
    #files.append(glob.glob(args.inDir + '*/'+ tracks[i] + '/all_flav*.root')[0])
    #files.append(glob.glob(args.inDir +'*'+sampleType + '*/'+ tracks[i] + '/all_flav*EMPf_*.root')[0])
    #else:
        #files = glob.glob(args.inDir + '*' + sampleType + '*/all_flav*.root')
        #files = glob.glob(args.inDir+'*'+sampleType+'*/*/all_flav*.root')

    tracks = args.tracks.split(':')
    fileDict = {}
    for track in tracks:
        #if args.tracks == "": track = files[i].split('_')[-1].replace('.root','')
        fileDict[track] = glob.glob(args.inDir + '*' + sampleType + '*/' + track + '/*/flav_Akt4EMPf.root')
        print fileDict[track]

        jetVars = {
            'jet_LabDr_HadF': [],  # b = 5, c = 4, l = 0
            #'jet_jf_n2t': [],
            #'jet_jf_m': [],
            #'jet_jf_sig3d': [],
            #'jet_jf_efc': [],
            #'jet_jf_nvtx1t': [],
            #'jet_jf_nvtx': [],
            #'jet_jf_ntrkAtVx': [],
            #'jet_jf_dR': [],
            #'jet_dRiso': [], # to categorise isojets
            #'jet_pt': [],
            'jet_jf_llr': [],

        }

        vtxVars = {
            'jet_LabDr_HadF': [],  # to categorise jets
            'jet_jf_nvtx': [],
            'jet_jf_nvtx1t': [],
            'jet_jf_n2t': [],
            'jet_jf_sig3d': [],
            'jet_jf_vtx_ntrk': [],
            'jet_jf_vtx_L3D': [],
            'jet_jf_vtx_sig3D': [],
            'jet_jf_ntrkAtVx': [],
            'jet_trk_jf_Vertex': [],
            'jet_jf_vtx_chi2': [],
        }

        trackVars = {
            'jet_trk_truthMatchProbability': [],
            'jet_trk_pt': [],
            'jet_trk_jf_Vertex': [],
            'jet_LabDr_HadF': [], # b = 5, c = 4, l = 0
            'jet_dRiso': [], # to categorise isojets
            'jet_pt': [],
            'jet_trk_orig': [], # PUFAKE = -1, FROMB = 0, FROMC = 1, FRAG = 2, GEANT = 3
            'jet_jf_llr': [],
            'jetEventNumber': [],
        }

        if args.jetVars == False: jetVars = ""
        if args.trackVars == False: trackVars = ""

        p = multiprocessing.Process(target=saveDictionaries, args=(fileDict[track], args.outDir, args.version, track,
                                                                   jetVars, trackVars, "", ""))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()