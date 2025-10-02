import ROOT as M
import numpy as np
import pandas as pd
import sys

# Load MEGAlib into ROOT
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

# Initialize MEGAlib
G = M.MGlobal()
G.Initialize()

GeometryName = sys.argv[1]
Filename = sys.argv[2]
output_file = sys.argv[3]
shared = sys.argv[4]
noise = sys.argv[5]

print("Ligth curve loaded !")

# Load geometry:
Geometry = M.MDGeometryQuest()
if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
  print("Geometry " + GeometryName + " loaded!")
else:
  print("Unable to load geometry " + GeometryName + " - Aborting!")
  quit()

if noise=="0":
    Geometry.ActivateNoising(False)

Reader = M.MFileEventsSim(Geometry)

if Reader.Open(M.MString(Filename)) == False:
    print("Unable to open file " + Filename + ". Aborting!")
    quit()

#per hit 
BGO_Z0_0 = 0.
BGO_Z0_1 = 0.
BGO_Z0_2 = 0.
BGO_Z0_3 = 0.
BGO_Z0_4 = 0.
BGO_Z1_4 = 0.
BGO_Z1_3 = 0.
BGO_Z1_2 = 0.
BGO_Z1_1 = 0.
BGO_Z1_0 = 0.

BGO_X1_0 = 0.
BGO_X1_1 = 0.
BGO_X1_2 = 0.

BGO_X0_0 = 0.
BGO_X0_1 = 0.
BGO_X0_2 = 0.

BGO_Y1_0 = 0.
BGO_Y1_1 = 0.
BGO_Y1_2 = 0.

BGO_Y0_0 = 0.
BGO_Y0_1 = 0.
BGO_Y0_2 = 0.

time = 0.
latX = 0.
lonX = 0.
latZ = 0.
lonZ = 0.

#per event
Time = []
Energy_Z1 = []
Energy_Z0 = []
Energy_X1 = []
Energy_X0 = []
Energy_Y1 = []
Energy_Y0 = []
LatX = []
LonX = []
LatZ = []
LonZ = []

FirstHit = True

while True:
    Event = Reader.GetNextEvent()

    if not Event:
        break
    M.SetOwnership(Event, True)
    
    if Event.GetNIAs() > 0:
    
        for i in range(Event.GetNHTs()):
            Hit = Event.GetHTAt(i)
            if Hit.GetDetectorType() == 8:
                
                pos = Hit.GetPosition()
                 
                x=pos.X()
                y=pos.Y()
                z=pos.Z()
                
                detector = Geometry.GetDetector(pos).GetName()

                
                #bottom
                if detector.GetString() == "BGO_Z0_0":
                    BGO_Z0_0+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Z0_1":
                    BGO_Z0_1+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Z0_2":
                    BGO_Z0_2+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Z0_3":
                    BGO_Z0_3+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Z0_4":
                    BGO_Z0_4+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Z1_4":
                    BGO_Z1_4+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Z1_3":
                    BGO_Z1_3+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Z1_2":
                    BGO_Z1_2+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Z1_1":
                    BGO_Z1_1+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Z1_0":
                    BGO_Z1_0+=(Hit.GetEnergy())
                    
                
                #Y pannel
                elif detector.GetString() == "BGO_Y1_0":
                    BGO_Y1_0+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Y1_1":
                    BGO_Y1_1+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Y1_2":
                    BGO_Y1_2+=(Hit.GetEnergy())
                
                #Y neg pannel
                elif detector.GetString() == "BGO_Y0_0":
                    BGO_Y0_0+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Y0_1":
                    BGO_Y0_1+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_Y0_2":
                    BGO_Y0_2+=(Hit.GetEnergy())

                #X pannel
                elif detector.GetString() == "BGO_X1_0":
                    BGO_X1_0+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_X1_1":
                    BGO_X1_1+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_X1_2":
                    BGO_X1_2+=(Hit.GetEnergy())
                
                #X neg pannel
                elif detector.GetString() == "BGO_X0_0":
                    BGO_X0_0+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_X0_1":
                    BGO_X0_1+=(Hit.GetEnergy())
                elif detector.GetString() == "BGO_X0_2":
                    BGO_X0_2+=(Hit.GetEnergy())
                else:
                    print(detector)
                    print(str(x)+" "+str(y)+" "+str(z))
                    print("coordinate not found")
                    sys.exit()
		
                if FirstHit:
		    #time
                    time = ( Event.GetTime().GetAsSeconds())

                    # x axis of space craft pointing at GAL latitude
                    latX=(np.float32(Event.GetGalacticPointingXAxisLatitude()))
                
                    # x axis of space craft pointing at GAL longitude
                    lonX=(np.float32(Event.GetGalacticPointingXAxisLongitude()))
                
                    # z axis of space craft pointing at GAL latitude
                    latZ=(np.float32(Event.GetGalacticPointingZAxisLatitude()))
                
                    # z axis of space craft pointing at GAL longitude
                    lonZ=(np.float32(Event.GetGalacticPointingZAxisLongitude()))

                    FirstHit = False
            
    
    #bot neg
    if BGO_Z0_0 >= 80. and BGO_Z0_0 <= 2000.:
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(BGO_Z0_0))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)        
    if BGO_Z0_1 >= 80. and BGO_Z0_1 <= 2000.:
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(BGO_Z0_1))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    if BGO_Z0_2 >= 80. and BGO_Z0_2 <= 2000.:
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(BGO_Z0_2))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    if BGO_Z0_3 >= 80. and BGO_Z0_3 <= 2000.:
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(BGO_Z0_3))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
        
        
    if BGO_Z0_4 >= 80. and BGO_Z0_4 <= 2000.:
        Energy_Z1.append(np.float32(0))
        
        if shared == "true":
            Energy_Y0.append(np.float32(BGO_Z0_4))  ## ASIC shared
            Energy_Z0.append(np.float32(0))
        else:
            Energy_Z0.append(np.float32(BGO_Z0_4))
            Energy_Y0.append(np.float32(0))
            
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0)) 
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)

    #bot
    if BGO_Z1_4 >= 80. and BGO_Z1_4 <= 2000.:
        
        if shared == "true":
            Energy_Y1.append(np.float32(BGO_Z1_4)) ## ASIC shared
            Energy_Z1.append(np.float32(0))
        else:
            Energy_Z1.append(np.float32(BGO_Z1_4))
            Energy_Y1.append(np.float32(0))
            
        Energy_Z0.append(np.float32(0))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y0.append(np.float32(0))  
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
        
    if BGO_Z1_3 >= 80. and BGO_Z1_3 <= 2000.:
        Energy_Z0.append(np.float32(0))
        Energy_Z1.append(np.float32(BGO_Z1_3))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    if BGO_Z1_2 >= 80. and BGO_Z1_2 <= 2000.:
        Energy_Z0.append(np.float32(0))
        Energy_Z1.append(np.float32(BGO_Z1_2))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    if BGO_Z1_1 >= 80. and BGO_Z1_1 <= 2000.:
        Energy_Z0.append(np.float32(0))
        Energy_Z1.append(np.float32(BGO_Z1_1))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    if BGO_Z1_0 >= 80. and BGO_Z1_0 <= 2000.:
        Energy_Z0.append(np.float32(0))
        Energy_Z1.append(np.float32(BGO_Z1_0))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)

    if BGO_X1_0 >= 80. and BGO_X1_0 <= 2000.:
        Energy_X1.append(np.float32(BGO_X1_0))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)

    if BGO_X1_1 >= 80. and BGO_X1_1 <= 2000.:
        Energy_X1.append(np.float32(BGO_X1_1))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)

    if BGO_X1_2 >= 80. and BGO_X1_2 <= 2000.:
        Energy_X1.append(np.float32(BGO_X1_2))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)

    if BGO_Y1_0 >= 80. and BGO_Y1_0 <= 2000.:
        Energy_Y1.append(np.float32(BGO_Y1_0))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_X1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    
    if BGO_Y1_1 >= 80. and BGO_Y1_1 <= 2000.:
        Energy_Y1.append(np.float32(BGO_Y1_1))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_X1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    
    if BGO_Y1_2 >= 80. and BGO_Y1_2 <= 2000.:
        Energy_Y1.append(np.float32(BGO_Y1_2))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_X1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)

    if BGO_X0_0 >= 80. and BGO_X0_0 <= 2000.:
        Energy_X0.append(np.float32(BGO_X0_0))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_X1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)

    if BGO_X0_1 >= 80. and BGO_X0_1 <= 2000.:
        Energy_X0.append(np.float32(BGO_X0_1))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_X1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)

    if BGO_X0_2 >= 80. and BGO_X0_2 <= 2000.:
        Energy_X0.append(np.float32(BGO_X0_2))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_X1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)

    if BGO_Y0_0 >= 80. and BGO_Y0_0 <= 2000.:
        Energy_Y0.append(np.float32(BGO_Y0_0))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    
    if BGO_Y0_1 >= 80. and BGO_Y0_1 <= 2000.:
        Energy_Y0.append(np.float32(BGO_Y0_1))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    
    if BGO_Y0_2 >= 80. and BGO_Y0_2 <= 2000.:
        Energy_Y0.append(np.float32(BGO_Y0_2))
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)


    BGO_Z0_0 = 0.
    BGO_Z0_1 = 0.
    BGO_Z0_2 = 0.
    BGO_Z0_3 = 0.
    BGO_Z0_4 = 0.
    BGO_Z1_4 = 0.
    BGO_Z1_3 = 0.
    BGO_Z1_2 = 0.
    BGO_Z1_1 = 0.
    BGO_Z1_0 = 0.
    BGO_X1_0 = 0.
    BGO_X1_1 = 0.
    BGO_X1_2 = 0.
    BGO_X0_0 = 0.
    BGO_X0_1 = 0.
    BGO_X0_2 = 0.
    BGO_Y1_0 = 0.
    BGO_Y1_1 = 0.
    BGO_Y1_2 = 0.
    BGO_Y0_0 = 0.
    BGO_Y0_1 = 0.
    BGO_Y0_2 = 0.

    Event = 0
    FirstHit = True


df = pd.DataFrame({"timestamp[s]": Time,"bgo_z1[keV]":Energy_Z1,"bgo_z0[keV]":Energy_Z0
               ,"bgo_x1[keV]":Energy_X1,"bgo_x0[keV]":Energy_X0,
               "bgo_y1[keV]":Energy_Y1,"bgo_y0[keV]":Energy_Y0,
               "latX":LatX,"lonX":LonX,"latZ":LatZ,"lonZ":LonZ })

df['datetime']= pd.to_datetime(df["timestamp[s]"],unit="s")
df.set_index("datetime",inplace=True)

if shared == "true":
    output_file = output_file.replace(".evt","_shared.evt")
df.to_csv(output_file, index=True) 

# create file with counts for each of the 6 panels

bgo_data_seconds = {}

for col in df.columns:
    if col.startswith("bgo_"):
        filtered_data = df[df[col] > 0][[col]]
        times_in_seconds = filtered_data.index.map(lambda x: x.timestamp())
        bgo_data_seconds[col] = np.array(times_in_seconds)


output_file_lc = open(output_file+".lc","w")

for col, times_in_seconds in bgo_data_seconds.items():

    output_file_lc.write(col+" "+str(len(times_in_seconds))+"\n")
    
output_file_lc.close()
 