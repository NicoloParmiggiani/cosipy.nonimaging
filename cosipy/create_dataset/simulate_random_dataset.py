"""
Simulate random GRB dataset for COSI
====================================

This script generates a simulated GRB dataset for COSI by creating, for each
direction in an HEALPix grid (defined by nside), a job directory containing a
MEGAlib source file (GRB.source) and a SLURM script (job.slurm), and submits the
simulations via Docker. You can control the number of replicas per pixel, the
random seed, the average photon flux (fixed value or uniform range), and the
photon spectrum (predefined or random choice).

Inputs (flags):
- --path-analysis: base directory where job folders are created.
- --run-name: run name used to build paths inside Docker.
- --grbs-per-pixel: number of GRBs to simulate per HEALPix pixel; -1 simulates one GRB per pixel (default: -1).
- --seed: random seed; -1 to generate a different seed per job (default: -1).
- --nside: HEALPix resolution (required).
- --random-flux: average photon flux; accepts "min:max" for a uniform draw or a single value.
- --random-spectrum: "yes" to pick a random predefined spectrum, or 0/1/2 to select a specific predefined spectrum (default: "0").
- --noise: string passed to post-processing (e.g., "true"/"false").
- --max-job: maximum number of concurrent jobs in the SLURM queue (default: 100).
- --geometry-path: path to the MEGAlib geometry .geo.setup file (container path).
 - --limit-grb-number: sample N coordinates (GRBs) from the grid; -1 uses all (default: -1).
 - --path-repository: host absolute path to mount at /deeplearning_cosi (default: /cosi-grb/codidl/).
 - --path-data: host absolute path to mount at /data (default: /cosi-grb/data).

Outputs:
- One directory per job: {theta}_{phi}_{seed} inside `--path-analysis`,
  containing `GRB.source`, `job.slurm`, and the generated `.sim`/`.evt` files.

Example:
python3 simulate_random_dataset.py --path-analysis="/data/analysis/run1" --run-name="run1" \\
  --nside=8 --grbs-per-pixel=1 --seed=-1 --random-flux="1:30" --random-spectrum="yes" \\
  --limit-grb-number=100 \\
  --geometry-path="/data/models/massmodel-cosi-smex-detailed/COSISMEX.sim_BGOreading.geo.setup" \
  --path-repository="/cosi-grb/codidl/" \
  --path-data="/cosi-grb/data" \
  --noise="true" --max-job=200
"""

import os,random,subprocess
import argparse
import numpy as np
import healpy as hp
import time
import csv
import yaml

parser = argparse.ArgumentParser(description="Generate and submit GRB simulations on an HEALPix grid")
parser.add_argument('--path-analysis', dest='path_analysis', required=True, help='Base directory where job folders are created')
parser.add_argument('--run-name', dest='run_name', required=True, help='Run name used to build container paths')
parser.add_argument('--grbs-per-pixel', dest='grbs_per_pixel', type=int, default=-1, help='Number of GRBs to simulate per HEALPix pixel; -1 uses one job per pixel')
parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Global random seed; -1 to generate a different one per job')
parser.add_argument('--nside', dest='nside', type=int, required=True, help='HEALPix resolution')
parser.add_argument('--random-flux', dest='random_flux', required=True, help='Average photon flux: single value or range "min:max"')
parser.add_argument('--random-spectrum', dest='random_spectrum', default='0', help='"yes" random predefined, or fixed index 0/1/2')
parser.add_argument('--noise', dest='noise', required=True, help='Noise parameter for post-processing (e.g., true/false)')
parser.add_argument('--max-job', dest='max_job', type=int, default=100, help='Maximum number of simultaneous jobs in SLURM queue')
parser.add_argument('--geometry-path', dest='geometry_path', required=True, help='Path to the MEGAlib geometry .geo.setup file (container path)')
parser.add_argument('--limit-grb-number', dest='limit_grb_number', type=int, default=-1, help='Sample N coordinates from the grid; -1 uses all')
parser.add_argument('--path-repository', dest='path_repository', default='/cosi-grb/codidl/', help='Host absolute path to mount at /deeplearning_cosi')
parser.add_argument('--path-data', dest='path_data', default='/cosi-grb/data', help='Host absolute path to mount at /data')

args = parser.parse_args()

path_analysis = args.path_analysis
run_name = args.run_name
grbs_per_pixel = args.grbs_per_pixel
seed = args.seed
nside = args.nside
random_flux = args.random_flux
random_spectrum = args.random_spectrum
noise = args.noise
max_job = args.max_job
geometry_path = args.geometry_path
limit_grb_number = args.limit_grb_number
path_repository = args.path_repository
path_data = args.path_data

# Create an HEALPix grid of coordinates given a specific nside density
# Return two arrays: [[l1,l2,l3...],[b1,b2,b3...]]

def ring_list(nside=0):
    npix_healpix = hp.nside2npix(nside)

    theta_ring_healpix = np.zeros(npix_healpix)
    phi_ring_healpix = np.zeros(npix_healpix)
    
    for jp in range(npix_healpix):
        theta_rad, phi_rad = hp.pix2ang(nside, jp, nest=True)
        theta_deg = np.degrees(theta_rad)
        phi_deg = np.degrees(phi_rad)
        theta_ring_healpix[jp] = theta_deg
        phi_ring_healpix[jp] = phi_deg
    
    return theta_ring_healpix, phi_ring_healpix


coords_hp = ring_list(nside)

coordinates_list  = []
for i in range(0,len(coords_hp[0])):

    if grbs_per_pixel!=-1:
        for j in range(0,grbs_per_pixel):
            coordinates_list.append((coords_hp[0][i],coords_hp[1][i]))
    else:
        coordinates_list.append((coords_hp[0][i],coords_hp[1][i]))
        
coordinates = np.array(coordinates_list)


if(limit_grb_number!=-1):
    print("sample from coordinate")
    indices = np.random.choice(coordinates.shape[0], size=limit_grb_number, replace=False)
    coordinates = coordinates[indices]

print("Array of Elements:")
print(coordinates.shape)

num_coordinates = coordinates.shape[0]

for coord in coordinates:
    
    theta= round(coord[0],3)
    phi = round(coord[1],3)

    print("create job for "+str(theta)+" "+str(phi))
    #create job
    exist = True
    while(exist==True):

        if(seed==-1):
            random_seed = random.randint(1, 100000)
        else:
            random_seed = seed

        dir_name=path_analysis+"/"+str(theta)+"_"+str(phi)+"_"+str(random_seed)

        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print("exist")
            exist=True
        else:
            exist=False

        dir_name_docker = "/data/analysis/"+run_name+"/"+str(theta)+"_"+str(phi)+"_"+str(random_seed)

    os.mkdir(dir_name)
    os.chdir(dir_name)
    
    if ":" in random_flux:
        min_flux = float(random_flux.split(":")[0])
        max_flux = float(random_flux.split(":")[1])
        flux = np.random.uniform(min_flux, max_flux)
    else:
        flux = random_flux

    spectra_list = ["Band 10 10000 -1.9 -3.7 230","Band 10 10000 -1 -2.3 699.9","Comptonized 10 10000 -0.5 1500"]
    
    if random_spectrum == "yes":
        spectra = random.choice(spectra_list)
    else:
        spectra = spectra_list[int(random_spectrum)]
             
    source_file = """
        # Global Parameters
        Version                     1
        Geometry                    """+geometry_path+"""

        # Physics list
        PhysicsListEM               LivermorePol

        # Output formats
        StoreSimulationInfo         init-only

        # Store shield counts
        PreTriggerMode              EveryEventWithHits

        # Run and source parameters
        Run                         GRBSim
        GRBSim.FileName             GRB
        GRBSim.Time                 1.0
        GRBSim.Source               GRB
        GRB.ParticleType    1
        GRB.Beam            FarFieldPointSource """+str(theta)+""" """+str(phi)+"""

        # Spectrum 
        GRB.Spectrum        """+spectra+""" 

        # Average photon flux in photon/cm2/s 
        GRB.Flux            """+str(flux)+"""
        """

    file_source = open("GRB.source","w")
    file_source.write(source_file)
    file_source.close()
    
    job_content =  """#!/bin/bash
        #SBATCH --partition=large
        #SBATCH --job-name=sim_cosi_grb

        docker run --user "$(id -u)" --entrypoint "" --rm -v """+path_data+""":/data -v /data01:/data01 -v /data02:/data02 -v """+path_repository+""":/deeplearning_cosi cosi_mega_bctools:v1.0.4_parmiggiani /bin/bash -i -c 'cd """+dir_name_docker+""" && cosima -u -s """+str(random_seed)+""" GRB.source'
        
        docker run --user "$(id -u)" --entrypoint "" --rm -v """+path_data+""":/data -v /data01:/data01 -v /data02:/data02 -v """+path_repository+""":/deeplearning_cosi cosi_mega_bctools:v1.0.4_parmiggiani /bin/bash -i -c 'cd """+dir_name_docker+""" && python3 /deeplearning_cosi/create_dataset/read_sim_files.py """+geometry_path+""" """+dir_name_docker+"""/*sim """+dir_name_docker+"""/"""+str(theta)+"""_"""+str(phi)+"""_shared.evt true """+noise+"""' 
        
        docker run --user "$(id -u)" --entrypoint "" --rm -v """+path_data+""":/data -v /data01:/data01 -v /data02:/data02 -v """+path_repository+""":/deeplearning_cosi cosi_mega_bctools:v1.0.4_parmiggiani /bin/bash -i -c 'cd """+dir_name_docker+""" && python3 /deeplearning_cosi/create_dataset/read_sim_files.py """+geometry_path+""" """+dir_name_docker+"""/*sim """+dir_name_docker+"""/"""+str(theta)+"""_"""+str(phi)+""".evt false """+noise+"""'"""
      
    file_job = open("job.slurm","w")
    file_job.write(job_content)
    file_job.close()
        

    os.system("sbatch job.slurm") #-o /dev/null -e /dev/null  

    # Wait until all jobs complete
    result = subprocess.run("squeue --format=%j", shell=True, text=True, capture_output=True)

    array = result.stdout.split("\n")
    job_number = len(array)-2
    while job_number> max_job:
        time.sleep(0.01)
        result = subprocess.run("squeue --format=%j", shell=True, text=True, capture_output=True)
        array = result.stdout.split("\n")
        job_number = len(array)-2

    

    
    
