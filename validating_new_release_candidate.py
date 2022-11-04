import os
import pandas as pd
import numpy as np
from scipy import stats
import re
import tarfile
import holoviews as hv
            
import pickle
import tarfile
import numpy as np
import itertools

hv.extension('bokeh')


from_timestamp = lambda x, dtype='datetime64[ms]': np.array(x*1000).astype(dtype)

pd.set_option('display.max_columns', 200)

colors = ['#2E73E2', '#CC9933', '#33CC99', '#9933CC', '#343435', '#000066']



def import_1ksps_file(filepath, save_file=True):
    data = []
    with tarfile.open(filepath, "r") as zf:
        zfp = zf.extractfile(zf.getnames()[0])
        try:
            while True:
                data.append(pickle.load(zfp))
        except EOFError as e:
            print(e)

    pressure_a = []
    pressure_b = []
    speed = []
    for pkt in data:
        pressure_a.append(pkt[0])
        pressure_b.append(pkt[1])
        if len(pkt[2]):
            speed.append(pkt[2])

    print('File read finished')

    pressure_a = list(zip(*pressure_a))
    pressure_a_time_posix = np.array(list(itertools.chain(*pressure_a[0])))
    pressure_a_bar = np.array(list(itertools.chain(*pressure_a[1])))

    pressure_b = list(zip(*pressure_b))
    pressure_b_time_posix = np.array(list(itertools.chain(*pressure_b[0])))
    pressure_b_bar = np.array(list(itertools.chain(*pressure_b[1])))

    if len(speed):
        speed = list(zip(*speed))
        speed_time = np.array(list(itertools.chain(*speed[0])))
        speed_rpm = np.array(list(itertools.chain(*speed[1])))
    else:
        speed_time = []
        speed_rpm = []
        
    output = (pressure_a_time_posix, pressure_a_bar, pressure_b_time_posix, pressure_b_bar, speed_time, speed_rpm)

    print('Array construction finished')

    if save_file*0:
        with open('{}.pickle'.format(filename_no_ext), 'wb') as fp:
            pickle.dump(output, fp)                       
    return output


if __name__ == "__main__":

    mount_point='/mnt/g'
    root='Shared drives/300 - Engineering/Trucks/dataCollected'
    data_folder = 'CB20300035'
    files_ = os.listdir(os.path.join(mount_point, root, data_folder))

    files = [f for f in files_ if 'tar.gz' in f]
    # for file in files:
    file = files[2]
    filepath = os.path.join(mount_point,root, data_folder, file)
    data = import_1ksps_file(filepath, save_file=False)

    o = []
    for t, value in zip(data[0::2], data[1::2]):
        o.append(hv.Curve((from_timestamp(t), value)).opts(alpha=0.5)*hv.Scatter((from_timestamp(t), value)).opts(width=600))

    display(hv.Overlay(o))
