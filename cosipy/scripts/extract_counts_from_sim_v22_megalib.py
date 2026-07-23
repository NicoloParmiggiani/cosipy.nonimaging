### Copyright Nicolò Parmiggiani
### script created starting from Savitri Gallego code

import ROOT as M
import numpy as np
import pandas as pd
import sys


#M.gRandom.SetSeed(111)
# Load MEGAlib into ROOT
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

# Initialize MEGAlib
G = M.MGlobal()
G.Initialize()

# We are good to go ...

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

#by default Megalib is noising the energy. Uncomment below if you want to assume perfect energy resolution
#Geometry.ActivateNoising(False);
if noise=="0":
    Geometry.ActivateNoising(False)

Reader = M.MFileEventsSim(Geometry)

#Reader.ShowProgress()
if Reader.Open(M.MString(Filename)) == False:
    print("Unable to open file " + Filename + ". Aborting!")
    quit()

#per hit 
ACS_Z0_0 = 0.
ACS_Z0_1 = 0.
ACS_Z0_2 = 0.
ACS_Z0_3 = 0.
ACS_Z0_4 = 0.
ACS_Z1_4 = 0.
ACS_Z1_3 = 0.
ACS_Z1_2 = 0.
ACS_Z1_1 = 0.
ACS_Z1_0 = 0.

ACS_X1_0 = 0.
ACS_X1_1 = 0.
ACS_X1_2 = 0.

ACS_X0_0 = 0.
ACS_X0_1 = 0.
ACS_X0_2 = 0.

ACS_Y1_0 = 0.
ACS_Y1_1 = 0.
ACS_Y1_2 = 0.

ACS_Y0_0 = 0.
ACS_Y0_1 = 0.
ACS_Y0_2 = 0.

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

event_number = 0


while True:
    Event = Reader.GetNextEvent()

    if not Event:
        break
    M.SetOwnership(Event, True)
    
    event_number = event_number +1
    
    if Event.GetNIAs() > 0:
    
        for i in range(Event.GetNHTs()):
            Hit = Event.GetHTAt(i)
            if Hit.GetDetectorType() == 8:
                
                pos = Hit.GetPosition()
                 
                x=pos.X()
                y=pos.Y()
                z=pos.Z()
                
                detector_object = Geometry.GetDetector(pos)

                if not detector_object:
                    print(f"Detector not found at position: {x} {y} {z}")
                    sys.exit(1)

                detector = detector_object.GetName()

                
                #bottom
                if detector.GetString() == "ACS_Z0_0":
                    ACS_Z0_0+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Z0_1":
                    ACS_Z0_1+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Z0_2":
                    ACS_Z0_2+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Z0_3":
                    ACS_Z0_3+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Z0_4":
                    ACS_Z0_4+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Z1_4":
                    ACS_Z1_4+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Z1_3":
                    ACS_Z1_3+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Z1_2":
                    ACS_Z1_2+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Z1_1":
                    ACS_Z1_1+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Z1_0":
                    ACS_Z1_0+=(Hit.GetEnergy())
                    
                
                #Y pannel
                elif detector.GetString() == "ACS_Y1_0":
                    ACS_Y1_0+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Y1_1":
                    ACS_Y1_1+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Y1_2":
                    ACS_Y1_2+=(Hit.GetEnergy())
                
                #Y neg pannel
                elif detector.GetString() == "ACS_Y0_0":
                    ACS_Y0_0+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Y0_1":
                    ACS_Y0_1+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_Y0_2":
                    ACS_Y0_2+=(Hit.GetEnergy())

                #X pannel
                elif detector.GetString() == "ACS_X1_0":
                    ACS_X1_0+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_X1_1":
                    ACS_X1_1+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_X1_2":
                    ACS_X1_2+=(Hit.GetEnergy())
                
                #X neg pannel
                elif detector.GetString() == "ACS_X0_0":
                    ACS_X0_0+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_X0_1":
                    ACS_X0_1+=(Hit.GetEnergy())
                elif detector.GetString() == "ACS_X0_2":
                    ACS_X0_2+=(Hit.GetEnergy())
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
    if ACS_Z0_0 >= 80. and ACS_Z0_0 <= 2000.:
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(ACS_Z0_0))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)        
    if ACS_Z0_1 >= 80. and ACS_Z0_1 <= 2000.:
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(ACS_Z0_1))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    if ACS_Z0_2 >= 80. and ACS_Z0_2 <= 2000.:
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(ACS_Z0_2))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    if ACS_Z0_3 >= 80. and ACS_Z0_3 <= 2000.:
        Energy_Z1.append(np.float32(0))
        Energy_Z0.append(np.float32(ACS_Z0_3))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
        
        
    if ACS_Z0_4 >= 80. and ACS_Z0_4 <= 2000.:
        Energy_Z1.append(np.float32(0))
        
        if shared == "true":
            Energy_Y0.append(np.float32(ACS_Z0_4))  ## ASIC shared
            Energy_Z0.append(np.float32(0))
        else:
            Energy_Z0.append(np.float32(ACS_Z0_4))
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
    if ACS_Z1_0 >= 80. and ACS_Z1_0 <= 2000.:
        
        if shared == "true":
            Energy_Y1.append(np.float32(ACS_Z1_0)) ## ASIC shared
            Energy_Z1.append(np.float32(0))
        else:
            Energy_Z1.append(np.float32(ACS_Z1_0))
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
        
    if ACS_Z1_3 >= 80. and ACS_Z1_3 <= 2000.:
        Energy_Z0.append(np.float32(0))
        Energy_Z1.append(np.float32(ACS_Z1_3))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    if ACS_Z1_2 >= 80. and ACS_Z1_2 <= 2000.:
        Energy_Z0.append(np.float32(0))
        Energy_Z1.append(np.float32(ACS_Z1_2))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    if ACS_Z1_1 >= 80. and ACS_Z1_1 <= 2000.:
        Energy_Z0.append(np.float32(0))
        Energy_Z1.append(np.float32(ACS_Z1_1))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)
    if ACS_Z1_4 >= 80. and ACS_Z1_4 <= 2000.:
        Energy_Z0.append(np.float32(0))
        Energy_Z1.append(np.float32(ACS_Z1_4))
        Energy_X1.append(np.float32(0))
        Energy_X0.append(np.float32(0))
        Energy_Y1.append(np.float32(0))
        Energy_Y0.append(np.float32(0))
        Time.append(time)
        LatX.append(latX)
        LonX.append(lonX)
        LatZ.append(latZ)
        LonZ.append(lonZ)

    if ACS_X1_0 >= 80. and ACS_X1_0 <= 2000.:
        Energy_X1.append(np.float32(ACS_X1_0))
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

    if ACS_X1_1 >= 80. and ACS_X1_1 <= 2000.:
        Energy_X1.append(np.float32(ACS_X1_1))
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

    if ACS_X1_2 >= 80. and ACS_X1_2 <= 2000.:
        Energy_X1.append(np.float32(ACS_X1_2))
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

    if ACS_Y1_0 >= 80. and ACS_Y1_0 <= 2000.:
        Energy_Y1.append(np.float32(ACS_Y1_0))
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
    
    if ACS_Y1_1 >= 80. and ACS_Y1_1 <= 2000.:
        Energy_Y1.append(np.float32(ACS_Y1_1))
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
    
    if ACS_Y1_2 >= 80. and ACS_Y1_2 <= 2000.:
        Energy_Y1.append(np.float32(ACS_Y1_2))
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

    if ACS_X0_0 >= 80. and ACS_X0_0 <= 2000.:
        Energy_X0.append(np.float32(ACS_X0_0))
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

    if ACS_X0_1 >= 80. and ACS_X0_1 <= 2000.:
        Energy_X0.append(np.float32(ACS_X0_1))
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

    if ACS_X0_2 >= 80. and ACS_X0_2 <= 2000.:
        Energy_X0.append(np.float32(ACS_X0_2))
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

    if ACS_Y0_0 >= 80. and ACS_Y0_0 <= 2000.:
        Energy_Y0.append(np.float32(ACS_Y0_0))
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
    
    if ACS_Y0_1 >= 80. and ACS_Y0_1 <= 2000.:
        Energy_Y0.append(np.float32(ACS_Y0_1))
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
    
    if ACS_Y0_2 >= 80. and ACS_Y0_2 <= 2000.:
        Energy_Y0.append(np.float32(ACS_Y0_2))
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


    ACS_Z0_0 = 0.
    ACS_Z0_1 = 0.
    ACS_Z0_2 = 0.
    ACS_Z0_3 = 0.
    ACS_Z0_4 = 0.
    ACS_Z1_4 = 0.
    ACS_Z1_3 = 0.
    ACS_Z1_2 = 0.
    ACS_Z1_1 = 0.
    ACS_Z1_0 = 0.
    ACS_X1_0 = 0.
    ACS_X1_1 = 0.
    ACS_X1_2 = 0.
    ACS_X0_0 = 0.
    ACS_X0_1 = 0.
    ACS_X0_2 = 0.
    ACS_Y1_0 = 0.
    ACS_Y1_1 = 0.
    ACS_Y1_2 = 0.
    ACS_Y0_0 = 0.
    ACS_Y0_1 = 0.
    ACS_Y0_2 = 0.

    Event = 0
    FirstHit = True


df = pd.DataFrame({"timestamp[s]": Time,"ACS_z1":Energy_Z1,"ACS_z0":Energy_Z0
               ,"ACS_x1":Energy_X1,"ACS_x0":Energy_X0,
               "ACS_y1":Energy_Y1,"ACS_y0":Energy_Y0,
               "latX":LatX,"lonX":LonX,"latZ":LatZ,"lonZ":LonZ })

df['datetime']= pd.to_datetime(df["timestamp[s]"],unit="s")
df.set_index("datetime",inplace=True)

if shared == "true":
    output_file = output_file.replace(".evt","_shared.evt")
df.to_csv(output_file, index=True) 

# create file with coutns for each of the 6 panels

ACS_data_seconds = {}

for col in df.columns:
    if col.startswith("ACS_"):
        filtered_data = df[df[col] > 0][[col]]
        times_in_seconds = filtered_data.index.map(lambda x: x.timestamp())
        ACS_data_seconds[col] = np.array(times_in_seconds)


output_file_lc = open(output_file+".lc","w")

for col, times_in_seconds in ACS_data_seconds.items():

    output_file_lc.write(col+" "+str(len(times_in_seconds))+"\n")
    
output_file_lc.close()
 