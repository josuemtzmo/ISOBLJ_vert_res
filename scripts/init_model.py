from MITgcmutils import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import *
import gsw 

if len(sys.argv) < 2:
    raise ValueError("Missing vertical resolution, run python init_model.py 1")

# Model grid dimensions
dx = 60 # m
dy = 60 # m
dz = int(sys.argv[1]) # m

nx = 20*3
ny = 20*8
ncores = nx*ny
print("Ncores:", ncores/20**2)

nz = int(512/dz)

# Path to store the input.
input_path = "/data/hpcdata/users/josnez/input/ISOBLJ/expt_dz{0}_z512/".format(dz)

# Dimensions as in the input
delX = nx*dx
delY = ny*dy
delR = nz*dz

# Create coordinates
x = np.linspace(0,delX,nx)
y = np.linspace(0,delY,ny)
z = np.linspace(0,delR,nz)

#################################################################
################ EXPERIMENT CONFIGURATION #######################
#################################################################

m_slopes = np.degrees(np.arctan(np.array([16/delX,32/delX,64/delX,96/delX,128/delX])))


# Generate slopes for the experiments:
m_slopes       = [32/delX,64/delX,96/delX,128/delX,96/delX,96/delX,
                  96/delX,96/delX, 96/delX,96/delX]
n_slopes       = [ 0/delY, 0/delY, 0/delY,  0/delY, 0/delY, 0/delY,
                   0/delY, 0/delY,  0/delY, 0/delY]
gaus_amplitude = [     80,     80,     80,      80,   160,     120,
                       60,     40,      20,      0]
thermal_drive = [2,1,0.5,0.1]

# CDW like near PIG based on Pierre Dutrieux et al., (2014). DOI:10.1126/science.1244341
Sref = 34.5     
#################################################################

# Create empty shapes to populate with data.
zeros_surface = np.zeros((nx,ny))
empty_domain = np.ones((nz,nx,ny))

##################################################################
###################### VEL & Pressure ############################
##################################################################

print("Writting velocities and pressure.")

# Create and save velocity and pressure fields:
vel_mag = 5e-4
u,v = np.random.uniform(low=-1.0, high=1.0, size=(2,nz,nx,ny)) * vel_mag
# Write u
writebin(input_path+"ini_uvel.bin", u, dataprec="float64")
# Write v
writebin(input_path+"ini_vvel.bin", v, dataprec="float64")
# Write p
writebin(input_path+"ini_shice_p.bin", zeros_surface, dataprec="float64")

##################################################################
########################## SLOPES ################################
##################################################################

print("Writting bathymetry and ice shelf topography.")

if len(m_slopes) != len(n_slopes):
    raise ValueError("Lengths of slope arrays sould be equal")

shift_z = 16
shift_dz = 8
    
for i, slope in enumerate(m_slopes):
    m = m_slopes[i]
    n = n_slopes[i]

    vshift_match = (np.max(gaus_amplitude) - gaus_amplitude[i])
    max_depth_all_sims = np.max(gaus_amplitude) + shift_z + 96
    
    if m != 96/delX:
        vshift_match = ( ( max_depth_all_sims - gaus_amplitude[i] ) + np.min(slope_function(x, y, m, n, d= - 16 , stype="surf"))  )

    surface_slope = slope_function(x, y, m, n, d= - shift_z - vshift_match , stype="surf")
    bottom_slope = slope_function(x, y, m, n, d=-delR + shift_dz)

    vshiftx,vshifty = shift_x_y(m, n, delX, delY, dz)
    sx, sy = slope_percentage(delX,delY,surface_slope.T)
    print("Slope percentages:", sx,sy)
    print("Slope angles:", np.degrees(np.arctan(m)),np.degrees(np.arctan(n)))

    sigma = 1200/3
    
    gaussian_channel = gaussian2D(x,y,gaus_amplitude[i],delX/2,delY/2,
                                  sigma_x=sigma, sigma_y=sigma)
    gaussian_channel = np.round(gaussian_channel - gaussian_channel.max(),shift_dz)
    
    # Make sure that gaussians match.
    if abs(gaussian_channel.min()) != gaus_amplitude[i]:
        raise ValueError("""There is a mismatch between the prescribed 
                            and generated gaussian amplitude.""")

    channel_surface = surface_slope + gaussian_channel

    if abs(np.min(channel_surface)) != abs(max_depth_all_sims):
        raise ValueError("The max depth doesn't match between simulations.")

    
    
    # print(np.unique(bottom_slope - channel_surface))
    # Write bathymetry
    output_file = input_path+"bathy_channel_vshift{0}_c{1}m_dz{2}.bin"
    writebin(output_file.format(vshiftx,gaus_amplitude[i],dz), 
             bottom_slope, dataprec="float64")
    # Write iceshelf topography
    output_file = input_path+"ini_shice_topo_channel_vshift{0}_c{1}m_dz{2}.bin"
    writebin(output_file.format(vshiftx,gaus_amplitude[i],dz), 
             channel_surface, dataprec="float64")

##################################################################
######################### RESTORING ##############################
##################################################################

print("Writting restoring mask.")

for i, slope in enumerate(m_slopes):

    m = m_slopes[i]
    n = n_slopes[i]

    vshiftx,vshifty = shift_x_y(m, n, delX, delY, dz)
    
    bottom_slope = slope_function(x, y, m, n, d=-delR + shift_dz)
    sx, sy = slope_percentage(delX,delY,bottom_slope.T)
    
    depth = np.cumsum(empty_domain,axis=0) * dz
    mask = mask_depth(depth, bottom_slope.T, delta=-shift_z,  sigma = 20)
    mask = np.transpose(mask,(0,2,1))

    # Enforce values to be 0 or 1 when the mask values are small (~1%).
    mask[mask<=0.01]=0
    mask[mask>=0.99]=1
    # Write masks
    writebin(input_path+"rbcs_mask_vshift{0}.bin".format(vshiftx), mask, dataprec="float64")

print("Writting restoring field.")

surface_slope = slope_function(x, y, np.max(m_slopes), 
                               np.max(n_slopes), d=-shift_z, stype="surf")

flat_surface = surface_slope - np.max(gaus_amplitude)

g = 9.81
rho_i = 917.0
p_pascal = g * rho_i * abs(flat_surface)
p_dbar = 10 * ( p_pascal/100000 )


a0 = -0.0573 
a1 =  0.0    
a2 =  0.0    
c0 =  0.0832 
b  =  -7.61e-4

pLoc = 300

freezing_point = a0*Sref+c0+b*pLoc

print(freezing_point)
#freezing_point = -2.12195 #np.min(gsw.t_freezing(Sref,p_dbar,saturation_fraction=0))

Tref = np.array([freezing_point + therD  for therD in thermal_drive])
print(Tref)


for ti, Temp in enumerate(Tref):
    tdstr = str(thermal_drive[ti]).replace(".","")
    restore_T = np.round(Temp,5) * empty_domain
    restore_S = np.round(Sref,5) * empty_domain

    t0 = str(round(abs(Temp),2)).replace('.','_')
    # Write rbcs salt
    writebin(input_path+"rbcs_temp_t{0}.bin".format(tdstr), restore_T, dataprec="float64")
    # Write rbcs salt
    writebin(input_path+"rbcs_salt_t{0}.bin".format(tdstr), restore_S, dataprec="float64")




    
