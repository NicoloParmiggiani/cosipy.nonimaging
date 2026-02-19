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
    for col in df.columns:
        if col.startswith("SCB"):
            filtered_data = df_cut[df_cut[col] > 0][[col]]
            times_in_seconds = filtered_data.index.to_numpy(dtype=float)
            bgo_times[col] = np.array(times_in_seconds)
            
            
    extracted_data = {'z1':bgo_times['SCB2-A1[keV]'],
                      'z0':bgo_times['SCB2-A0[keV]'],
                          'x1':bgo_times['SCB0-A1[keV]'],
                          'x0':bgo_times['SCB0-A0[keV]'],
                          'y1':bgo_times['SCB1-A1[keV]'],
                          'y0':bgo_times['SCB1-A0[keV]']}
    
    return extracted_data

if data_challenge == "DC3":

    data_preprocessed = open_and_read_csv_counts_DC3(file_path)

elif data_challenge == "DC4":
    
    data_preprocessed = open_and_read_csv_counts_DC4(file_path)
    
else:
    print("Invalid data challenge selected")

np.savez(output_path
        ,z1=data_preprocessed['z1']
        ,z0=data_preprocessed['z0']
        ,x1=data_preprocessed['x1']
        ,x0=data_preprocessed['x0']
        ,y1=data_preprocessed['y1']
        ,y0=data_preprocessed['y0'])

    

    