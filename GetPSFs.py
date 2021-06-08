# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:30:56 2021

@author: David Nguyen
"""


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


# Read binary files from SIMVIEW
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
def gauss_3d(coords, bg, A_coeff, x_0, y_0, z_0, x_sig, y_sig, z_sig):
    # Expose X, Y, and Z coordinates
    z_val, x_val, y_val = coords
    
    # General Gaussian function 3D
    return bg + A_coeff * np.exp(-((x_val-x_0) ** 2 / (2 * x_sig ** 2) +
                                   (y_val-y_0) ** 2 / (2 * y_sig ** 2) +
                                   (z_val-z_0) ** 2 / (2 * z_sig ** 2)))


def main():
    # Read beads data
    img_path = r'E:\SIMVIEW 5 Data\0.1um-beads_CloseToDet2_20210603_151041\SPC00_TM00000_ANG000_CM0_CHN00_PH0.stack'
    (beads, (size_x, size_y, size_z)) = read_binary(img_path)

    # Data type boundaries
    type_max = np.iinfo(beads.dtype).max
    type_min = np.iinfo(beads.dtype).min
       
    # Remove saturated data
    beads[beads == type_max] = type_min

    # Bounding box
    x_box = 4
    y_box = 4
    z_box = 14
    
    # Grids
    x_val = np.array(range(0, x_box+x_box+1))
    y_val = np.array(range(0, y_box+y_box+1))
    z_val = np.array(range(0, z_box+z_box+1))
    z_grid, x_grid, y_grid = np.meshgrid(z_val, x_val, y_val, indexing='ij')
    xdata = np.vstack((z_grid.ravel(), x_grid.ravel(), y_grid.ravel()))
    
    # Initial guess
    bg = 1e3
    A_coeff = 1e4
    x_0 = x_box
    y_0 = y_box
    z_0 = z_box
    x_sig = 1
    y_sig = x_sig
    z_sig = 2
    p0 = bg, A_coeff, x_0, y_0, z_0, x_sig, y_sig, z_sig
    
    # Loop parameters
    loop = True
    count = 0
    threshold = 500
    
    # Initializations
    x_coord_abs = np.zeros(threshold,)
    y_coord_abs = np.zeros(threshold,)
    z_coord_abs = np.zeros(threshold,)
    
    x_sigma = np.zeros(threshold,)
    y_sigma = np.zeros(threshold,)
    z_sigma = np.zeros(threshold,)
    
    RMSE = np.zeros(threshold,)
    
    while loop:
        # Find first maximum in stack
        z_coord, x_coord, y_coord = np.unravel_index(np.argmax(beads,
                                                               axis=None),
                                                     beads.shape)

        # Index in range? Otherwise put to type_min (edge cases)
        x_limits = x_coord > x_box and x_coord < size_x-1-x_box
        y_limits = y_coord > y_box and y_coord < size_y-1-y_box 
        z_limits = z_coord > z_box and z_coord < size_z-1-z_box 
            
        if x_limits and y_limits and z_limits:
            # Retrieve bead bounding box (the box doesn't have ownership in
            # Python, if beads is modified, box will be modified as well)
            box = beads[z_coord-z_box:z_coord+z_box+1,
                        x_coord-x_box:x_coord+x_box+1,
                        y_coord-y_box:y_coord+y_box+1]
            
            # Check all voxels of the box > type_min (valid bead data)
            if np.amin(box) > type_min:
                # 3D Gaussian fit
                popt, pcov = curve_fit(gauss_3d, xdata, box.ravel(), p0)
                
                x_coord_abs[count] = popt[2] + x_coord - x_box
                y_coord_abs[count] = popt[3] + y_coord - y_box
                z_coord_abs[count] = popt[4] + z_coord - z_box
                x_sigma[count] = popt[5]
                y_sigma[count] = popt[6]
                z_sigma[count] = popt[7]
                
                RMSE[count] = mean_squared_error(gauss_3d(xdata, *popt),
                                                 box.ravel())
                RMSE[count] = math.sqrt(RMSE[count])
                
                count += 1 
                if count >= threshold:
                    loop = False   
                    
            # Clear box
            beads[z_coord-z_box:z_coord+z_box,
                  x_coord-x_box:x_coord+x_box,
                  y_coord-y_box:y_coord+y_box] = type_min
        else:
            beads[z_coord, x_coord, y_coord] = type_min
    
    return x_coord_abs, y_coord_abs, z_coord_abs, x_sigma, y_sigma, z_sigma, RMSE


if __name__ == "__main__":
    x_coord_abs, y_coord_abs, z_coord_abs, x_sigma, y_sigma, z_sigma, RMSE = main()
     
    det = 2304
    pix = 6.5
    mag = 20
    
    z_step = 1
    
    Y_axis = ( y_coord_abs - det / 2 ) * pix / mag
    
    plt.figure()
    plt.scatter(Y_axis, np.abs( 2.355 * z_sigma * z_step ), s=10,
                c=np.log(RMSE/np.amax(RMSE)*100), cmap='cividis', alpha=0.5)
    ax = plt.axes()
    ax.set_facecolor("darkgray")
    plt.grid(color='gray')
    plt.xlabel('Y position [um] (direction of light-sheet propagation)')
    plt.ylabel('Axial FWHM [um]')
    plt.ylim(0, 10)
    cbar = plt.colorbar()
    cbar.set_label('Normalized RMSE in log-scale')
    plt.show()
    
    # plt.figure()
    # plt.hist(x_sigma)
    # plt.show()
    
    # plt.figure()
    # plt.hist(y_sigma)
    # plt.show()
    
    # plt.figure()
    # plt.hist(z_sigma)
    # plt.show()
    
    # plt.figure()
    # plt.hist(RMSE)
    # plt.show()