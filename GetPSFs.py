# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:30:56 2021

@author: David Nguyen
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Read binary files from SiMView
def read_binary(file_path, size_x=2304, size_y=2304,
                        file_type=np.uint16):
    # Read binary stack as a single vector of 16-bits unsigned integers
    int_stack = np.fromfile(file_path, dtype=file_type)

    # Determine Z size automatically based on the array size
    size_z = int(int_stack.size/size_x/size_y)

    # Reshape the stack based on known dimensions
    int_stack = np.reshape(int_stack, (size_z, size_y, size_x))
    
    # Return the stack
    return (int_stack, (size_x, size_y, size_z))


# Gaussian function to fit on 3D bead data
def gauss_3d(coords, bg, A_coeff, x_0, y_0, z_0,
             x_sig, y_sig, z_sig):
    z_val, x_val, y_val = coords
    return bg + A_coeff * np.exp(-((x_val-x_0) ** 2 / (2 * x_sig ** 2) +
                                   (y_val-y_0) ** 2 / (2 * y_sig ** 2) +
                                   (z_val-z_0) ** 2 / (2 * z_sig ** 2)))


def main():
    # Read beads data
    img_path = r'E:\Data\20210316_0.1um-beads\_20210316_131127\SPC00_TM00000_ANG000_CM0_CHN00_PH0.stack'
    (beads, (size_x, size_y, size_z)) = read_binary(img_path)

    # Data type boundaries
    type_max = np.iinfo(beads.dtype).max
    type_min = np.iinfo(beads.dtype).min
       
    # Remove saturated data
    beads[beads == type_max] = type_min
    
    # Bounding box
    x_pos = 4
    x_neg = 4
    y_pos = 4
    y_neg = 4
    z_pos = 14
    z_neg = 14
    
    # Grids
    x_val = np.array(range(0, x_pos+x_neg+1))
    y_val = np.array(range(0, y_pos+y_neg+1))
    z_val = np.array(range(0, z_pos+z_neg+1))
    z_grid, x_grid, y_grid = np.meshgrid(z_val, x_val, y_val, indexing='ij')
    xdata = np.vstack((z_grid.ravel(), x_grid.ravel(), y_grid.ravel()))
    
    # Initial guess
    bg = 1e3
    A_coeff = 1e4
    x_0 = x_neg
    y_0 = y_neg
    z_0 = z_neg
    x_sig = 2
    y_sig = x_sig
    z_sig = 4
    p0 = bg, A_coeff, x_0, y_0, z_0, x_sig, y_sig, z_sig
    
    loop = True
    count = 0
    threshold = 100
    
    x_sigma = np.zeros(threshold,)
    y_sigma = np.zeros(threshold,)
    z_sigma = np.zeros(threshold,)
    
    while loop:
        # Find first maximum in stack
        indices = np.where(beads == np.amax(beads))
        x_coord = indices[1][0]
        y_coord = indices[2][0]
        z_coord = indices[0][0]

        # Index in range? Otherwise put to type_min (edge cases)
        x_limits = x_coord > x_neg and x_coord < size_x-1-x_pos
        y_limits = y_coord > y_neg and y_coord < size_y-1-y_pos 
        z_limits = z_coord > z_neg and z_coord < size_z-1-z_pos 
            
        if x_limits and y_limits and z_limits:
            # Retrieve bead bounding box (the box doesn't have ownership in
            # Python, if beads is modified, box will be modified as well)
            box = beads[z_coord-z_neg:z_coord+z_neg+1,
                        x_coord-x_neg:x_coord+x_pos+1,
                        y_coord-y_neg:y_coord+y_neg+1]
            
            # Check all voxels of the box > type_min (valid bead data)
            if np.amin(box) > type_min:
                popt, pcov = curve_fit(gauss_3d, xdata, box.ravel(), p0)
                x_sigma[count] = popt[5]
                y_sigma[count] = popt[6]
                z_sigma[count] = popt[7]
                
                count += 1 
                if count >= threshold:
                    loop = False   
                    
            # Clear box
            beads[z_coord-z_neg:z_coord+z_neg,
                  x_coord-x_neg:x_coord+x_pos,
                  y_coord-y_neg:y_coord+y_neg] = type_min
        else:
            beads[z_coord, x_coord, y_coord] = type_min

    plt.hist(x_sigma)
    plt.hist(y_sigma)
    plt.hist(z_sigma)


if __name__ == "__main__":
     main()