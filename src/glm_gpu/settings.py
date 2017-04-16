import os
import inspect
import pyopencl as pcl

my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CL_FILE = os.path.join(my_location, 'cl', 'logistic.cl')

platform_index = 0
platform = pcl.get_platforms()[platform_index]
# TODO: set up a method to change this via user interface

devices = platform.get_devices()
for idx, d in enumerate(devices):
    if pcl.device_type.to_string(d.type) == 'GPU' and 'AMD Radeon' in d.name:
        device_index = idx
        break

CL_PREFERRED_DEVICE = (platform_index, device_index)
