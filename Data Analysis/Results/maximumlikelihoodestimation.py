from pathlib import Path
import numpy as np


if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    data_folder_path = PROJECT_ROOT / "Raw_Data/Position_Data/"
    txt_file_paths = list(data_folder_path.glob("data*.txt"))

    print("NUMBER OF FILES READING WITH 'data*.txt' STYLE FILENAME IN ", data_folder_path, " : ", 
          len(txt_file_paths))

    """
    We have the equation 
    (2Dt)_est = 1/2N ∑{from i=1 to i=N} (r_i^2)

    and the goal is to estimate D,
    where N is the number of samples (equal to len(txt_file_paths) here)
    and t is the number of timesteps.

    Short note on determining t: we filmed 2 FPS with 120 frames over 60 seconds (a minute); 
    Note time between each frame is therefore also ~0.5 ± 0.03 seconds
    """
    t = 60
    N = len(txt_file_paths)

    squared_distances = []

    for txt_file in txt_file_paths:
        data_arr = np.loadtxt(fname=txt_file,
                        delimiter="\t",
                        skiprows=2,
                        dtype="float64")
        #First column x values, second column y values
        #Conversion from px to μm => mult by 0.12048 ± 0.003 μm
        x = data_arr[:,0]*0.12048#*10**(-6)
        y = data_arr[:,1]*0.12048#*10**(-6)
        r_squared = np.square(x[1:]-x[:-1]) + np.square(y[1:]-y[:-1])
        #Do math to get the inner summation term
        sum_squared_dist = np.sum(r_squared)

        squared_distances.append(sum_squared_dist)

    RHS = np.sum(np.array(squared_distances)/(2*N))
    D_est = RHS/(2*t)
    print("D estimate is: ", D_est)
    
        