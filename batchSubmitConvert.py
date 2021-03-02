import os
import sys
import subprocess
import glob


src_dir = '/unix/atlasvhbb2/ava/DL1_framework/'
out_dir = '/unix/atlastracking/ava/convertedInputs/'
in_dir_zp = '/unix/atlasvhbb2/srettie/tracking/athenaOutputs_full_custom_ipxd/'
in_dir_ttbar = '/unix/atlastracking/srettie/grid_downloads_custom_ipxd/'
conversion_script = '/unix/atlasvhbb2/ava/DL1_framework/tools/convert_fromROOT.py'

samples = {
    'zp': 'mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.merge.AOD.e6928_e5984_s3126_r10201_r10210',
    'ttbar': 'mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.merge.AOD.e6337_e5984_s3126_r10201_r10210'
}

files_per_job = 200
for sample in samples.keys():
    if sample == 'zp': in_dir = in_dir_zp + samples[sample]
    elif sample == 'ttbar': in_dir = in_dir_ttbar + samples[sample]

    track = sys.argv[1]
    outDir = out_dir + samples[sample] + '/' + track
    in_dir += '/' + track
    if not (os.path.isdir(outDir)): os.makedirs(outDir)

    singularity_cmd = 'export XDG_RUNTIME_DIR=""; singularity exec --bind {},{},{} docker://gitlab-registry.cern.ch/mguth/umami:latest '.format(out_dir, in_dir, src_dir)
    if sample == 'zp': files = glob.glob('{}/trk_*/flav_Akt4EMPf.root'.format(in_dir, track))
    elif sample == 'ttbar': files = glob.glob('{}/user.*/*.root'.format(in_dir))

    input_files = ''
    for i, step in enumerate(range(0, len(files), files_per_job)):
        if step + files_per_job < len(files):
            job_files = files[step:step + files_per_job]
        else:
            job_files = files[step:len(files)]

        script_name = '{}_{}_{}_h5.sh'.format(sample, track, i)
        w = open(src_dir + script_name, "w+")
        w.write('#! /bin/bash \n' + singularity_cmd)

        input_files = ''
        for f in job_files:
            input_files += ' ' + f

        python_cmd = 'python {} --output {} --input{} --write_tracks'.format(conversion_script,  outDir, input_files)
        w.write(python_cmd)
        w.close()

        cmd = 'qsub -q long -N ' + script_name + ' -j oe ' + src_dir + script_name
        subprocess.call(cmd, shell=True)
 
