import sys
import uproot as up
import numpy as np
import h5py
import pandas as pd
from Variable_mapping import *
from MV2_defaults import default_values2, track_defaults
import argparse
import datetime as dt
import glob
import multiprocessing
import itertools
import os

debug = True
tree_name = 'bTag_AntiKt4EMPFlowJets'#'bTag_AntiKt4EMTopoJets'


def GetArgs():

    """parse arguments"""

    parser = argparse.ArgumentParser(
        description="ROOT to hdf5 converter",
        usage="python convert_fromROOT.py <options>"
        )

    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "--input", action="store", dest="input_files",
        metavar="input files", required=True, nargs='+',
        help="full path and filename to input ROOT files to merge, using bash wildcards to select multiple files"
    )

    required.add_argument(
        "--output", action="store", dest="output",
        metavar="[path/to/filename]", required=True,
        help="path to output hdf5 file. Do not include a trailing slash --> /"
        )

    optional = parser.add_argument_group("optional arguments")

    optional.add_argument(
        "--events", action="store", dest="events", type=int,
        required=False, default=1e6, help="Amount of events to process"
        )

    optional.add_argument(
        "--single-file", action="store_true", dest="single",
        help="Option to save all events in one file"
        )
    optional.add_argument(
        "--track_type", action="store", dest="track_type", type=str,
        required=False, default="nom", help="Type of track to use"
    )
    optional.add_argument(
        "--suffix", action="store", dest="suffix", type=str,
        required=False, default="", help="Suffix for output file"
    )
    optional.add_argument(
        '--write_tracks', action='store_true', dest='write_tracks', default=False,
        help="Write track information to output file"
    )
    split = parser.add_mutually_exclusive_group()
    split.add_argument('--even', action='store_true')
    split.add_argument('--odd', action='store_true')
    category = parser.add_mutually_exclusive_group()
    category.add_argument('--bjets', action='store_true')
    category.add_argument('--cjets', action='store_true')
    category.add_argument('--ujets', action='store_true')
    return parser.parse_args()


def FindCheck(jetVec):
    default_location = np.argwhere(np.isnan(jetVec))
    jet_feature_check = np.zeros(len(jetVec))
    jet_feature_check[default_location] = 1
    return jet_feature_check

def get_file_suffix(filename):
    # Get a meaningful suffix for output filename from input filename
    return '/'.join(filename.split('/')[-2:]).replace('.root','').replace('flav_Akt4EMPf','').replace('trk_','').strip('/').split('._')[-1]

def flatten(nested):
    # Flatten a nested array to one lower dimension
    return list(itertools.chain.from_iterable(nested))

def GetTree(file_name, add_cuts="", write_tracks=False):
    """Retrieves the events in the TTree with uproot and returns them as
    a pandas DataFrame."""
    if debug:
        t0_jets = dt.datetime.now()
        print('Start GetTree')
    var_list = list(mapping.keys())
    tree = up.open(file_name)[tree_name]

    if write_tracks:
        tracks_ndarray = GetTracks(tree)
    if debug:
        print('Getting tracks ndarray took a total of: {}'.format(dt.datetime.now()-t0_jets))
        t0_jets = dt.datetime.now()

    df = tree.pandas.df(var_list)
    if debug:
        print('Getting df with uproot took: {}'.format(dt.datetime.now()-t0_jets))
        t0_jets = dt.datetime.now()

    df['jet_bH_pt'] = df.apply(lambda row: max(row['jet_bH_pt'])[0], axis=1)
    df['jet_bH_pt'] = df['jet_bH_pt'].mask(df['jet_bH_pt'].lt(0), 0)  # Set all negative bH pt values to 0
    df['jetPtRank'] = df.groupby(level=0)['jet_pt'].rank(ascending=False) # Add jet pT rank
    # If jet_jf_dR is larger than 15, it was set to the "default" value of std::hypot(-11,-11), so set this to its actual default of -1
    df['jet_jf_dR'] = df['jet_jf_dR'].mask(df['jet_jf_dR'].gt(15), default_values2['jf_dR'][0])

    # Apply jet quality cuts
    df.query('jet_pt>20e3 & abs(jet_eta)<2.5 & (abs(jet_eta)>2.4 |\
                jet_pt>60e3 | jet_JVT>0.5) & (jet_aliveAfterOR ==True)', inplace=True)

    if add_cuts != "":
        df.query(add_cuts, inplace=True)
    
    if debug:
        print('Querying jets df took: {}'.format(dt.datetime.now()-t0_jets))
        t0_jets = dt.datetime.now()

    df.rename(index=str, columns=mapping, inplace=True)
    # changing eta to absolute eta
    df['absEta_btagJes'] = df['eta_btagJes'].abs()
    # Replacing default values with this synthax
    # df.replace({'A': {0: 100, 4: 400}})
    rep_dict = {}
    for key, val in default_values2.items():
        if key in list(var_conv_oldDl1.keys()):
            replacer = {}
            for elem in val:
                replacer[elem] = np.nan
            rep_dict[var_conv_oldDl1[key]] = replacer
    df.replace(rep_dict, inplace=True)

    # Generating default flags
    df['JetFitter_isDefaults'] = FindCheck(df['JetFitter_mass'].values)
    df['SV1_isDefaults'] = FindCheck(df['SV1_masssvx'].values)
    df['IP2D_isDefaults'] = FindCheck(df['IP2D_bu'].values)
    df['IP3D_isDefaults'] = FindCheck(df['IP3D_bu'].values)
    df['JetFitterSecondaryVertex_isDefaults'] = FindCheck(df['JetFitterSecondaryVertex_nTracks'].values)
    # rnnip default flag not necessary anymore
    df['rnnip_isDefaults'] = FindCheck(df['rnnip_pu'].values)

    if debug:
        print('Remaining jets columns took: {}'.format(dt.datetime.now()-t0_jets))
        t0_jets = dt.datetime.now()

    if write_tracks:
        return df, tracks_ndarray
    else:
        return df
    
def getdPhi(jphi, tphi):
    # Calculate delta phi, accounting the for the 2pi at the boundary
    dphi = tphi - jphi
    dphi[dphi >  np.pi] = dphi[dphi >  np.pi] - 2*np.pi
    dphi[dphi < -np.pi] = dphi[dphi < -np.pi] + 2*np.pi
    return dphi
    
def getdR(jeta, jphi, teta, tphi):
    '''
    Taken from: https://gitlab.cern.ch/atlas-flavor-tagging-tools/rnnip/-/blob/master/root_to_np.py
    Calculate dR, accounting for the 2pi difference b/w for phi b/w pi and -pi.

    Inputs:
    - jeta, jphi: (floats) for the jet axis direction
    - teta, tphi: (np.arrays) for the track dir

    Returns:
    - tdrs: An np array for the opening angles b/w the tracks and the jet axis.

    '''

    deta = teta - jeta
    dphi = getdPhi(jphi, tphi)
    tdrs = np.sqrt( deta**2 + dphi**2)
    
    return tdrs

def GetTracks(tree):
    """Retrieves track information from tree loaded from uproot."""
    
    if debug:
        t0 = dt.datetime.now()

    track_var_list = list(track_mapping.keys())
    track_var_list.remove('jet_trk_ip2d_grade')
    array = tree.arrays(track_var_list)

    ### Get indices to associate tracks to jets and apply baseline cuts
    jet_select_var = ['jet_pt', 'jet_eta', 'jet_JVT', 'jet_aliveAfterOR', 'jet_phi_orig', 'jet_eta_orig', 'jet_pt_orig',
                      'jet_btag_ntrk']
    jet_df = tree.pandas.df(jet_select_var)
    level0 = jet_df.index.get_level_values(0).astype(str)
    level1 = jet_df.index.get_level_values(1).map("{:02}".format).values.astype(
        str)  # Ensure 2 digits for  unique indices
    jet_df['eventJetIndex'] = (level0 + "." + level1).astype(float)
    jet_df = jet_df.reset_index(drop=True)

    jet_df.query('jet_pt>20e3 & abs(jet_eta)<2.5 & (abs(jet_eta)>2.4 |\
                 jet_pt>60e3 | jet_JVT>0.5) & (jet_aliveAfterOR ==True)', inplace=True)
    pass_baseline = jet_df.index.astype(int)
    jet_var = ['jet_phi_orig', 'jet_eta_orig', 'jet_pt_orig', 'eventJetIndex']

    # Flatten nested list 3D -> 2D and apply baseline cuts
    tmp_dict = collections.OrderedDict()
    for trk_var in track_var_list:
        tmp_dict[trk_var] = np.array(flatten(array[trk_var.encode()]))[pass_baseline]

    # Flatten to 1D and repeat values
    track_dict = collections.OrderedDict()
    for var in jet_var:
        track_dict[var] = np.repeat(jet_df[var].values, jet_df['jet_btag_ntrk'].values)
    for trk_var in track_var_list:
        track_dict[trk_var] = flatten(tmp_dict[trk_var])
    df = pd.DataFrame.from_dict(track_dict)
    if debug:
        print('\tUnrolling tracks took: {}'.format(dt.datetime.now() - t0))
        t0 = dt.datetime.now()
    print (df)

    # Fix z0 --> z0sintheta
    df['jet_trk_z0'] = df['jet_trk_z0'] * np.sin(df['jet_trk_theta'])

    # Apply track selection
    df.query('jet_trk_pt>1000 & abs(jet_trk_d0)<1 & abs(jet_trk_z0)<1 & \
            jet_trk_eta<2.5 & jet_trk_nPixHits>=1 & \
             (jet_trk_nPixHits+jet_trk_nSCTHits) >= 7 & \
             (jet_trk_nsharedPixHits + floor(jet_trk_nsharedSCTHits/2)) <= 1 & \
             (jet_trk_nPixHoles + jet_trk_nSCTHoles) <= 2 & \
             jet_trk_nPixHoles <= 1', inplace=True)
    if debug:
        print('\tApplying track selection took: {}'.format(dt.datetime.now()-t0))
        t0 = dt.datetime.now()

    # Update old IP3D default (-10) to new default
    df['jet_trk_ip3d_grade'].replace(-10, track_defaults['IP3D_grade'], inplace=True)
    # Add jet_trk_ip2d_grade as IP3D equivalent
    df['jet_trk_ip2d_grade'] = df['jet_trk_ip3d_grade']
    if debug:
        print('\tUpdating IPxD grades took: {}'.format(dt.datetime.now()-t0))
        t0 = dt.datetime.now()
    print (df)

    # Calculate the derived variables: deta/dphi/dr/ptfrac
    df['deta'] = df['jet_trk_eta'] - df['jet_eta_orig']
    df['dphi'] = getdPhi(df['jet_phi_orig'], df['jet_trk_phi'])
    df['dr'] = getdR(df['jet_eta_orig'], df['jet_phi_orig'], df['jet_trk_eta'], df['jet_trk_phi'])
    df['ptfrac'] = df['jet_trk_pt']/df['jet_pt_orig']
    if debug:
        print('\tDerived variables calculation took: {}'.format(dt.datetime.now()-t0))
        t0 = dt.datetime.now()

    df = df.sort_values(['eventJetIndex', 'jet_trk_ip3d_d0sig'], ascending=[True, False])
    if debug:
        print('\tApplying sorting took: {}'.format(dt.datetime.now() - t0))
        t0 = dt.datetime.now()
    print (df)

    # Rename columns
    df.rename(index=str, columns=track_mapping, inplace=True)
    if debug:
        print('\tRenaming columns took: {}'.format(dt.datetime.now() - t0))
        t0 = dt.datetime.now()
    print (df)

    all_track_vars = list(track_mapping.values()) + ['deta', 'dphi', 'dr', 'ptfrac']
    # Zip track variables together into tuples
    df['tuple'] = [tuple(x) for x in df.filter(all_track_vars, axis=1).values]
    df_final = df.groupby('eventJetIndex')['tuple'].apply(list)
    if debug:
        print('\tZipping variables to tuples took: {}'.format(dt.datetime.now()-t0))
        t0 = dt.datetime.now()
    print (df)

    df_final = df_final.reset_index()
    df_final.drop(columns=['eventJetIndex'], inplace=True)
    trackCount = df.groupby('eventJetIndex')['tuple'].count().tolist()
    #maxTracks = df.groupby('eventJetIndex')['tuple'].count().max()
    maxTracks = 40
    remaining = [maxTracks - count for count in trackCount]
    trackLists = df_final['tuple'].tolist()
    defaultTuple = tuple(list(track_defaults.values()))
    for i in range(0, len(trackLists)):
        if remaining[i] <= 0:
            trackLists[i] = trackLists[i][:maxTracks]
        else:
            trackLists[i] += list((defaultTuple,) * remaining[i])
    if debug:
        print('\tPadding took: {}'.format(dt.datetime.now() - t0))
        t0 = dt.datetime.now()
    print (df)

    # Save tuples as ndarray
    tracks_ndarray = np.array(trackLists,dtype=np.dtype(dtype_list))
    if debug:
        print('\tDumping zipped variables to ndarray took: {}'.format(dt.datetime.now()-t0))
        t0 = dt.datetime.now()
    print (tracks_ndarray[0])
    print (tracks_ndarray[1])
    print (tracks_ndarray[2])
    print (tracks_ndarray[3])

    return tracks_ndarray


def __run():
    args = GetArgs()
    t0_glob = dt.datetime.now()
    if debug:
        print('Arguments:')
        for arg in vars(args): print(arg, getattr(args, arg))
        print('')
    if not (os.path.isdir(args.output)): os.makedirs(args.output)
        
    events = 0
    df_out = None
    # additional cuts on eventnb and class label
    add_cuts = ""
    parity_cut = ""
    if args.even:
        add_cuts = "(eventnb % 2 == 0)"
    elif args.odd:
        add_cuts = "(eventnb % 2 == 1)"
    if args.bjets:
        parity_cut = "(jet_LabDr_HadF == 5)"
    elif args.cjets:
        parity_cut = "(jet_LabDr_HadF == 4)"
    elif args.ujets:
        parity_cut = "(jet_LabDr_HadF == 0)"
    if parity_cut != "":
        add_cuts += "& %s" % parity_cut

    for i, file in enumerate(args.input_files):
        sample_type = 'zp'
        if '427081' in file: sample_type = 'zp_extended'
        if '410470' in file: sample_type = 'ttbar'

        events_rest = int(args.events - events)
        if events_rest <= 0:
            break
        # print(events_rest, "events more to process")
        sys.stdout.write('\r')
        # the exact output you're looking for:
        j = (events + 1) / args.events
        sys.stdout.write("%i/%i  [%-20s] %d%%" % (events, args.events,
                                                  '='*int(20*j), 100*j))
        sys.stdout.flush()

        if args.write_tracks:
            df, tracks_ndarray = GetTree(file, add_cuts, args.write_tracks)
        else:
            df = GetTree(file, add_cuts)

        if args.single is False:
            suffix = get_file_suffix(file)
            outfile_name = '{}/{}_{}-{}.h5'.format(args.output, sample_type, args.track_type, suffix)
            print('Saving to output file: {}'.format(outfile_name))
            h5f = h5py.File(outfile_name, 'w')
            h5f.create_dataset('jets',
                            data=df.to_records(index=False)[:],compression='gzip')
            if args.write_tracks:
                h5f.create_dataset('tracks',
                                data=tracks_ndarray,compression='gzip')
            h5f.close()
        else:
            if df_out is None:
                df_out = df
            else:
                df_out = pd.concat([df_out, df])
        events += len(df)
    print("")
    if args.single:
        outfile_name = '{}/{}_{}-merged.h5'.format(args.output, sample_type, args.track_type)
        print('Saving to output file: {}'.format(outfile_name))
        h5f = h5py.File(outfile_name, 'w')
        h5f.create_dataset('jets', data=df_out.sample(frac=1).to_records(
            index=False)[:int(events)],compression='gzip')
        h5f.close()
    print('Entire conversion process took: {}'.format(dt.datetime.now()-t0_glob))    
    print('Conversion successful!')

__run()