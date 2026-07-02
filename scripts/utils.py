import numpy as np

def slope_function(x,y,m=0,n=0,d=0,stype=""):
    """
    Function to compute a general slope for any given plane.
    """
    if x.ndim == 1 and x.ndim == 1:
        X,Y = np.meshgrid(x,y)
    elif x.ndim == 2 and x.ndim == 2:
        X = x
        Y = y
    else:
        raise ValueError("x and y dimensions should be 1 or 2.")

    slope = m*X + n*Y
    if stype=="surf":
        d_shift = d - np.max(np.abs(slope))
    else:
        d_shift = d #+ 2*np.max(np.abs(slope))
    z = d_shift + slope
    return z

def shift_x_y(m, n, delX, delY, dz):
    i_x = (delX*m)/dz
    i_y = (delY*n)/dz
        
    print("verShiftx={0}".format(int(i_x)))
    print("verShifty={0}".format(int(i_y)))
    return int(i_x),int(i_y)

def slope_percentage(delX,delY,slope):
    delta_slope = np.max(np.abs(slope),axis=0) - np.min(np.abs(slope),axis=0)
    slope_px = np.round(100*np.unique(delta_slope/delX)[0],2)
    delta_slope = np.max(np.abs(slope),axis=1) - np.min(np.abs(slope),axis=1)
    slope_py = np.round(100*np.unique(delta_slope/delY)[0],2)

    return slope_px, slope_py
    

def gaussian2D(x,y,A=1,x0=0,y0=0,sigma_x=1,sigma_y=1, a=0, b=1 ):
    """
    Function to compute a general slope for any given plane.
    """
    if x.ndim == 1 and x.ndim == 1:
        X,Y = np.meshgrid(x,y)
    elif x.ndim == 2 and x.ndim == 2:
        X = x
        Y = y
    else:
        raise ValueError("x and y dimensions should be 1 or 2.")
        
    z = np.exp(-( (a*(X-x0)**2)/(2*(sigma_x**2))  + (b*(Y-y0)**2)/(2*(sigma_y**2))  ) )
    z = A * (z - z.min())/ (z - z.min()).max()
    return z

def gaussian(x,x0=0,a=1,sigma=1):
    """
    Function to compute a general slope for any given plane.
    """
    
    z = np.exp(-( (a*(x-x0)**2)/(2*((sigma)**2)) ))
    return  z

def mask_depth(depth, bathy, delta, transition="smooth", amp=1, sigma=5):
    nz = depth.shape[0]
    nx = depth.shape[2]
    ny = depth.shape[1]
    mask = np.zeros_like(depth)
    for j in np.arange(ny):
        for i in np.arange(nx):
            if transition=="smooth":
                # idx = np.argmin(np.abs(-depth[:,i,j] - (bathy[i,j])))
                # print(np.linspace(0,-delR,nz), bathy[i,j] - delta)
                # mask[:,j,i] = smooth_tanh(-depth[:,j,i], bathy[j,i] - delta, amp, sigma)
                mask[:,j,i] = gaussian(-depth[:,j,i], bathy[j,i] - delta, amp, sigma)
                # break
            else:
                mask[:,j,i] = np.where(-depth[:,i,j]>=bathy[j,i]+delta,mask[:,i,j],1)
        #     break
        # break
        if transition=="smooth":
            mask = np.where( (-depth - (bathy - delta)) > 0, mask, 1)
    return mask

def smooth_tanh(x,a,b):
    return abs((0.5+0.5*np.tanh((x-a)/b)) - 1)
