#!/usr/bin/env python3

import time
import subprocess
from pathlib import Path
import os,sys

# -------- settings --------
CONFIG = sys.argv[1]
WORKDIR = Path(sys.argv[2])
NSIDE = 32

MAX_JOBS = 150
SLEEP = 0.01

JOB_NAME = "bc_rsp_pix"
CPUS = 1
MEM = "4G"
TIME = "04:00:00"
# --------------------------


NPIX = 12 * NSIDE**2


def active_jobs():
    cmd = ["squeue", "-h", "-n", JOB_NAME, "-t", "R,PD"]
    out = subprocess.run(cmd, text=True, capture_output=True).stdout
    return len(out.splitlines())


def pixel_done(pix):
    drm = WORKDIR / f"rsp_nside{NSIDE}_ring_pix{pix}" / "drm.h5"
    return drm.exists()


def submit_pixel(pix):
    script = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={MEM}
#SBATCH --time={TIME}
#SBATCH --output={WORKDIR}/slurm_pix_{pix}_%j.out
#SBATCH --error={WORKDIR}/slurm_pix_{pix}_%j.err

bc-rsp "{CONFIG}" "{WORKDIR}" \\
    --nparts {NPIX} \\
    --part {pix} \\
    --nthreads {CPUS}
"""

    result = subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
    )

    print(f"Submitted pixel {pix}: {result.stdout.strip()}")


for pix in range(NPIX):

    if pixel_done(pix):
        print(f"Pixel {pix} already done, skipping.")
        continue

    while active_jobs() >= MAX_JOBS:
        print(f"Too many jobs. Sleeping {SLEEP}s...")
        time.sleep(SLEEP)

    submit_pixel(pix)

print("All missing pixels submitted.")