import pandas as pd
import numpy as np
import os,sys

""" this is the mapping with the new geometry
    bgo_z1 = bottom_1_times
    bgo_z0 = bottom_2_times
    bgo_x0 = x1_times
    bgo_x1 = x2_times
    bgo_y1 = y1_times
    bgo_y0 = y2_times
"""

home = os.environ['HOME']

file_path = sys.argv[1]
output_path = sys.argv[2]
bin_size = float(sys.argv[3])
separate_panels = float(sys.argv[4])
ori_path = sys.argv[5]
data_challenge = sys.argv[6]

def open_and_read_csv_counts_DC3(file):
    
    col_names = [
        "Type",              
        "unix_time",         
        "x1", "x2", "x3",    
        "x4", "x5", "x6","x7",
        "SAA"            
    ]

    ori_df = pd.read_csv(
        ori_path,
        sep=r"\s+",         
        names=col_names,
        usecols=["unix_time", "SAA"],
        skiprows=1,              
        skip_blank_lines=True    
    )

    ori_df.set_index('unix_time', inplace=True)

    df = pd.read_csv(file, compression='gzip')
    
    df.set_index('timestamp[s]', inplace=True)
               
    df_tmp = ori_df.reindex(df.index,method="nearest")
    df["SAA"] = df_tmp["SAA"]
    df_cut = df[df["SAA"]!=0]
    
    bgo_times = {}
    for col in df_cut.columns:
        if col.startswith("bgo_"):
            filtered_data = df_cut[df_cut[col] > 0][[col]]
            times_in_seconds = filtered_data.index.to_numpy(dtype=float)
            bgo_times[col] = np.array(times_in_seconds)
            
            
    extracted_data = {'z1':bgo_times['bgo_bottom_1[keV]'],
                      'z0':bgo_times['bgo_bottom_2[keV]'],
                          'x0':bgo_times['bgo_x1[keV]'],
                          'x1':bgo_times['bgo_x2[keV]'],
                          'y0':bgo_times['bgo_y2[keV]'],
                          'y1':bgo_times['bgo_y1[keV]']}
                          
    
    return extracted_data

def open_and_read_csv_counts_DC4(file):

    print("DC4")

    col_names = [
        "Type", "unix_time",
        "x1", "x2", "x3", "x4", "x5", "x6", "x7",
        "SAA"
    ]

    # Read and clean the SAA reference file
    ori_df = pd.read_csv(
        ori_path,
        sep=r"\s+",
        names=col_names,
        usecols=["unix_time", "SAA"],
        skiprows=1,
        skip_blank_lines=True
    )

    ori_df[["unix_time", "SAA"]] = ori_df[
        ["unix_time", "SAA"]
    ].apply(pd.to_numeric, errors="coerce")

    ori_df = (
        ori_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["unix_time", "SAA"])
        .sort_values("unix_time")
        .drop_duplicates("unix_time")
        .set_index("unix_time")
    )

    # Read and clean the detector data
    df = pd.read_csv(file, compression="gzip")
    df["timestamp[s]"] = pd.to_numeric(
        df["timestamp[s]"],
        errors="coerce"
    )

    df = (
        df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["timestamp[s]"])
        .set_index("timestamp[s]")
    )

    scb_columns = [
        col for col in df.columns
        if col.startswith("SCB")
    ]

    df[scb_columns] = df[scb_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Assign the nearest SAA value to each detector timestamp
    df["SAA"] = ori_df.reindex(
        df.index,
        method="nearest"
    )["SAA"].to_numpy()

    df_cut = df[df["SAA"].notna() & df["SAA"].ne(0)]

    # Extract timestamps for events within the selected energy range
    bgo_times = {
        col: df_cut.index[
            df_cut[col].between(80.0, 2000.0, inclusive="both")
        ].to_numpy(dtype=float)
        for col in scb_columns
    }

    empty = np.array([], dtype=float)

    return {
        "z1": bgo_times.get("SCB2-A1[keV]", empty),
        "z0": bgo_times.get("SCB2-A0[keV]", empty),
        "x1": bgo_times.get("SCB0-A1[keV]", empty),
        "x0": bgo_times.get("SCB0-A0[keV]", empty),
        "y1": bgo_times.get("SCB1-A1[keV]", empty),
        "y0": bgo_times.get("SCB1-A0[keV]", empty)
    }

def open_and_read_csv_counts_DC4_v22(file):

    print("DC4_v22")

    # File di riferimento SAA
    ori_df = pd.read_csv(
        ori_path,
        sep=r"\s+",
        names=[
            "Type", "unix_time",
            "x1", "x2", "x3", "x4", "x5", "x6", "x7",
            "SAA"
        ],
        usecols=["unix_time", "SAA"],
        skiprows=1
    )

    ori_df[["unix_time", "SAA"]] = ori_df[
        ["unix_time", "SAA"]
    ].apply(pd.to_numeric, errors="coerce")

    ori_df = (
        ori_df.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("unix_time")
        .drop_duplicates("unix_time")
        .set_index("unix_time")
    )

    # File ACS
    df = pd.read_csv(file, compression="infer")

    acs_columns = [
        "ACS_z1", "ACS_z0",
        "ACS_x1", "ACS_x0",
        "ACS_y1", "ACS_y0"
    ]

    df[["timestamp[s]"] + acs_columns] = df[
        ["timestamp[s]"] + acs_columns
    ].apply(pd.to_numeric, errors="coerce")

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["timestamp[s]"])
        .sort_values("timestamp[s]")
        .set_index("timestamp[s]")
    )

    df["SAA"] = ori_df.reindex(
        df.index,
        method="nearest"
    )["SAA"].to_numpy()

    df = df[df["SAA"].notna() & df["SAA"].ne(0)]

    return {
        axis: df.index[
            df[f"ACS_{axis}"].between(80.0, 2000.0)
        ].to_numpy(dtype=float)
        for axis in ("z1", "z0", "x1", "x0", "y1", "y0")
    }
    
    
if data_challenge == "DC3":

    data_preprocessed = open_and_read_csv_counts_DC3(file_path)

elif data_challenge == "DC4":
    
    data_preprocessed = open_and_read_csv_counts_DC4(file_path)
    
elif data_challenge == "DC4_v22":
    
    data_preprocessed = open_and_read_csv_counts_DC4_v22(file_path)
    
else:
    print("Invalid data challenge selected")

np.savez(output_path
        ,z1=data_preprocessed['z1']
        ,z0=data_preprocessed['z0']
        ,x1=data_preprocessed['x1']
        ,x0=data_preprocessed['x0']
        ,y1=data_preprocessed['y1']
        ,y0=data_preprocessed['y0'])

    

    