import ROOT
import sys
from ROOT import *
import argparse
import multiprocessing
import glob
ROOT.gROOT.SetBatch()

parser = argparse.ArgumentParser(description='Get total number of jets')
parser.add_argument('-i', "--inDir", dest='inDir', default='/unix/atlasvhbb2/ava/FTAGFramework/athenaOutputs/',
                    help='input directory with merged nTuples from athena')
parser.add_argument('-t', "--tracks", dest='tracks', default='', help='input track collections')
parser.add_argument('-v', "--version", dest='version', default='427080_Zprime', help='input sample versions')
args = parser.parse_args()


def getTotalNumberFromCut(filepath, cut=""):
    file = ROOT.TFile(filepath)
    tree = file.Get("bTag_AntiKt4EMPFlowJets")
    cutString = TCut("(jet_LabDr_HadF == 5) || (jet_LabDr_HadF == 4) || (jet_LabDr_HadF == 0)")
    if cut != "": cutString += cut

    return tree.Draw(">>events", cutString, "entrylist")

def number(file, track):
    b = getTotalNumberFromCut(file, "jet_LabDr_HadF == 5")
    c = getTotalNumberFromCut(file, "jet_LabDr_HadF == 4")
    l = getTotalNumberFromCut(file, "jet_LabDr_HadF == 0")

    print track
    print b
    print c
    print l

if __name__ == "__main__":
    processes = []  # multiprocessing

    if '427081' in args.version: sampleType = 'Zprime_Extended'
    if '410470' in args.version: sampleType = 'ttbar'
    if '427080' in args.version: sampleType = 'Zprime'

    if args.tracks != "":
        if ':' not in args.tracks:
            tracks = []
            tracks.append(args.tracks)
        else:
            tracks = args.tracks.split(':')
        files = []
        for i in range(len(tracks)):
            # If specific folder, need to specify the sample type folder too
            # files.append(glob.glob(args.inDir + '*/'+ tracks[i] + '/all_flav*.root')[0])
            files.append(glob.glob(args.inDir + '*' + sampleType + '*/' + tracks[i] + '/all_flav*.root')[0])
    else:
        # files = glob.glob(args.inDir + '*' + sampleType + '*/all_flav*.root')
        files = glob.glob(args.inDir + '*' + sampleType + '*/*/all_flav*.root')

    for i in range(len(files)):
        if args.tracks == "":
            track = files[i].split('_')[-1].replace('.root', '')
        else:
            track = tracks[i]
        print track

        number (files[i], track)

