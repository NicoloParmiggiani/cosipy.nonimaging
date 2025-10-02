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

def open_and_read_csv_counts_DC3_newbkg(file):
    
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
            
            
    extracted_data = {'bottom_1_times':bgo_times['bgo_bottom_1[keV]'],
                      'bottom_2_times':bgo_times['bgo_bottom_2[keV]'],
                          'x1_times':bgo_times['bgo_x1[keV]'],
                          'x2_times':bgo_times['bgo_x2[keV]'],
                          'y1_times':bgo_times['bgo_y1[keV]'],
                          'y2_times':bgo_times['bgo_y2[keV]']}
    
    return extracted_data

data_preprocessed = open_and_read_csv_counts_DC3_newbkg(file_path)

np.savez(output_path
        ,bottom_1_times=data_preprocessed['bottom_1_times']
        ,bottom_2_times=data_preprocessed['bottom_2_times']
        ,x1_times=data_preprocessed['x1_times']
        ,x2_times=data_preprocessed['x2_times']
        ,y1_times=data_preprocessed['y1_times']
        ,y2_times=data_preprocessed['y2_times'])

    

    