import pyopencl as pcl
import os
import glob
import io


class OpenCLManager(object):

    def __init__(self,
                 cl_file,
                 preferred_device=None):

        if not os.path.isfile(cl_file) and not os.path.isdir(cl_file):
            raise ValueError("'cl_file' must be a directory with .cl files or one .cl file.")
        elif os.path.isfile(cl_file):
            fname, fext = os.path.splitext(cl_file)
            if fext.lower() != '.cl':
                raise TypeError("'cl_file' must have .CL or .cl as the file extension.")


        self.cl_file_path = cl_file
        # should be in the same format that Theano uses to designate devices eg 'opencln:n'
        # or it should be a tuple of (platform num, device num)
        if isinstance(preferred_device, str):
            pn_dn_str = preferred_device.lower().replace('opencl', '')
            pnum, dnum = [int(x) for x in pn_dn_str.split(":")]
        elif isinstance(preferred_device, tuple):
            pnum, dnum = preferred_device

        platform = pcl.get_platforms()[pnum]
        device = platform.get_devices()[dnum]

        self.device = device
        self.platform = platform

        # make a context and command queue
        self.context = pcl.Context([self.device])
        self.queue = pcl.CommandQueue(self.context)
        with io.open(self.cl_file_path, 'r') as f:
            self.programs = pcl.Program(self.context, f.read()).build()

        # make a dictionary of available kernels
        self.kernel_dictionary = {kernel.function_name:kernel for kernel in self.programs.all_kernels()}



    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_dev):
        if not isinstance(new_dev, pcl.Device):
            raise TypeError('{nd} is not a pyopencl.Device object'.format(nd=new_dev))
        else:
            self._device = new_dev

    @property
    def platform(self):
        return self._platform

    @platform.setter
    def platform(self, new_plat):
        if not isinstance(new_plat, pcl.Platform):
            raise TypeError('{np} is not a pyopencl.Platform object'.format(np=new_plat))
        else:
            self._platform = new_plat
