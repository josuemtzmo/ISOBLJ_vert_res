import xarray as xr
import numpy as np
import re


def shifted_computations(data, method="ud", dims=['Z']):
    def wrapped_shift(data, method):
        if method == "ud":
            prof = shift_profile_ud(data)
        elif method == "du":
            prof = shift_profile_du(data)
        return prof
    
    result = xr.apply_ufunc(
        wrapped_shift,
        data,
        kwargs={"method":method},
        input_core_dims=[dims],
        output_core_dims=[dims],
        dask='parallelized',
        vectorize=True,
        # exclude_dims = (),
        output_dtypes=[data.dtype]
    )
    
    return result

def shift_profile_du(data):
    value = data[0]
    if np.isnan(value):
        mask = np.where(np.isnan(data))[0]
    else:
        mask = np.where(data==data[0])[0]
    mask_diff = mask - np.roll(mask,-1)
    location_step = np.argmin(mask_diff) - np.min(mask_diff) - len(data)
    if np.min(mask_diff) == -1 :
        location_step = 0#len(mask_diff)
    data[mask] == np.nan
    return np.roll(data,-location_step)

def shift_profile_ud(data):
    value = data[0]
    if np.isnan(value):
        mask = np.where(np.isnan(data))[0]
    else:
        mask = np.where(data==data[0])[0]
    mask_diff = mask - np.roll(mask,-1)
    location_step = np.argmin(mask_diff) +1
    if np.min(mask_diff) == -1 :
        location_step = len(mask_diff)
    data[mask] == np.nan
    return np.roll(data,-location_step)

def shifted_dataset(dataset, method):
    dataset_shifted = dataset.copy()
    for var in dataset.data_vars:
        dims = [dims for dims in dataset[var].dims if "Z" in dims]
        dataset_shifted[var] = shifted_computations(dataset[var],method, dims=dims)

    return  dataset_shifted
    


import os
import glob
from xmitgcm import open_mdsdataset
import xgcm

class ISOBLJ_Struct():

    def __init__(self,folder,data=None):
        self.folder = folder
        if data is None:
            self.data = dict(
                dz8={"PATH":"run_z248_dz8/", "expt":{}},
                dz4={"PATH":"run_z248_dz4/", "expt":{}},
                dz2={"PATH":"run_z248_dz2/", "expt":{}},
                dz1={"PATH":"run_z248_dz1/", "expt":{}},
            )
        else:
            self.data = data
        self.search_experiments()

    def search_experiments(self):
            
        for key,item in self.data.items():
            folder_path = os.path.join(self.folder + self.data[key]["PATH"])
            for expt in os.listdir(folder_path):
                expt_path = folder_path+expt
                if os.path.isdir(expt_path):
                    expt = self._get_expt_dz_from_path(folder_path,expt)
                    item["expt"].update({expt:{}})
                    files = glob.glob(expt_path+"/*.*ta")
                    if len(files) !=0:
                        item["expt"][expt]["exec"]= True
                    else:
                        item["expt"][expt]["exec"]= False

                    nx,dx = self._get_value_from_data_file(expt_path+"/data", "delX")
                    ny,dy = self._get_value_from_data_file(expt_path+"/data", "delY")
                    nz,dz = self._get_value_from_data_file(expt_path+"/data", "delR")
                    
                    item["expt"][expt]['dims']={"nx":nx,"ny":ny,"nz":nz,
                                                "dx":dx,"dy":dy,"dz":dz
                                               }
                    dt = self._get_value_from_data_file(expt_path+"/data", "deltaT")
                    item["expt"][expt]['timestep']=dt


    def open_datasets(self,convert_units = True, gridargs={}, **kwargs):
        for key,item in self.data.items():
            folder_path = os.path.join(self.folder + self.data[key]["PATH"])
            for expt in os.listdir(folder_path):
                expt_path = folder_path+expt
                expt = self._get_expt_dz_from_path(folder_path,expt)
                if os.path.isdir(expt_path):
                    if item["expt"][expt]["exec"]:
                        dt = item["expt"][expt]["timestep"]
                        ds = open_mdsdataset(expt_path,delta_t=dt,**kwargs)
                        item["expt"][expt]['data']= ds
                        item["expt"][expt]['grid']= self.create_grid(ds,**gridargs)
                    else:
                        print("expt {0} has not run".format(expt))
        if convert_units:
            self._convert_units()
        
    def _get_grid(self):
        grid={}
        for key,item in self.data.items():
            for expt in item['expt'].keys():
                if item["expt"][expt]["exec"]:
                    grid[expt] = item['expt'][expt]['grid']
        return grid

    
    def _convert_units(self):
        for key,item in self.data.items():
            for expt in item['expt'].keys():
                if item["expt"][expt]["exec"]:
                    dataset = item["expt"][expt]['data']
                    coords = dataset.coords
                    for coor in coords.keys():
                        if "X" in coor or  "Y" in coor:
                            dataset[coor]=dataset[coor]/1000 # Convert to km
                        if "time" in coor: 
                            dataset[coor]=dataset[coor]/(60*60) # Convert to hrs
        

    @staticmethod
    def create_grid(dataset, **kwargs):
        return xgcm.Grid(dataset, **kwargs)

    def get_variable(self,var,silence=False):
        data = []
        rename_coord = []
        for key,item in self.data.items():
            for expt in item['expt'].keys():
                if item["expt"][expt]["exec"]:
                    if var not in item['expt'][expt]['data'].data_vars:
                        if silence==False:
                            print("{0} does not have var {1}".format(expt,var))
                        continue # Skip if variable is not found
                        
                    new_name = var+"_"+expt
                    dataset = item['expt'][expt]['data'][var].rename(new_name)

                    coords = dataset.coords
                    for coor in coords.keys():
                        if "Z" in dataset[coor].dims or "Zl" in dataset[coor].dims:
                            rename_coord.append(coor)
                    dz =  item['expt'][expt]['dims']['dz']                            
                    rename_dict = { coord: (coord+"_"+expt if "hFac" in coord  or "mask" in coord  
                                            else coord+"_dz{0:0.0f}".format(dz)) for coord in rename_coord}
                    data.append(dataset.rename(rename_dict))
        
            merge_ds = xr.merge(data,compat='override')
        return merge_ds

    # def get_hfacS(self,var):
    #     data = []
    #     rename_coord = []
    #     for key,item in self.data.items():
    #         for expt in item['expt'].keys():
    #             if item["expt"][expt]["exec"]:
    #                 new_name = var+"_"+expt
    #                 dataset = item['expt'][expt]['data'][var].rename(new_name)

    #                 coords = dataset.coords
    #                 for coor in coords.keys():
    #                     if "Z" in dataset[coor].dims or "Zl" in dataset[coor].dims:
    #                         rename_coord.append(coor)
    #                 dz =  item['expt'][expt]['dims']['dz']                            
    #                 rename_dict = { coord: (coord+"_"+expt+"_dz{0:0.0f}".format(8)  if "hFac" in coord  or "mask" in coord  
    #                                         else coord+"_dz{0:0.0f}".format(8)) for coord in rename_coord}
    #                 dataset = dataset.reset_coords([coord for coord in dataset.coords if "hfac" in coord]).rename(rename_dict)
    #                 dataset = dataset[[var for var in dataset.data_vars if "hfac" in var]]
    #                 # dataset = dataset.drop_vars([ coord for coord in rename_coord if coord not in dataset.dims])
    #                 data.append(dataset)
        
    #         merge_ds = xr.merge(data,compat='override')
    #     return merge_ds
        
    
    @staticmethod
    def _get_variable_of_experiment(var,expt):
        pass
    
    @staticmethod
    def _get_expt_dz_from_path(path,expt):
        dz = re.findall(r'\d+', path)[-1]
        return expt+"_dz"+dz        
                    
    @staticmethod
    def _get_string_from_data_file(datafile, strings):
        if type(strings) != list:
            strings = [strings]
            
        data=[]
        with open(datafile) as f:
            lines = f.readlines()
            for line in lines:
                if any(string in line for string in strings):
                    text = line.split('=')[1].replace(",","")
                    text = text.replace(" ","").replace("\n","")
                    data.append(text)
        if len(data) != len(strings):
            raise ValueError("Lengths of found matches does not match.")
        return data

    @staticmethod
    def _get_value_from_data_file(datafile, strings):
        data = []
        string_from_file = ISOBLJ_Struct._get_string_from_data_file(datafile, strings)
        for string in string_from_file:
            if "*" in string:
                n,value = np.array(string.split("*")).astype(float)
                data.append([n, value])
            else:
                value = np.array(string).astype(float)
                n=None
                data.append([ value])
                
        data=np.squeeze(data)
        return data