import pyopencl as pcl
import os
import glob


class OpenCLManager(object):

    def __init__(self,
                 cl_files_or_dir,
                 preferred_device=None):

        if not os.path.isfile(cl_files_or_dir) and not os.path.isdir(cl_files_or_dir):
            raise ValueError("'cl_files_or_dir' must be a directory with .cl files or one .cl file.")
        elif os.path.isfile(cl_files_or_dir):
            fname, fext = os.path.splittext(cl_files_or_dir)
            if fext.lower() != '.cl':
                raise TypeError("'cl_files_or_dir' must have .CL or .cl as the file extension.")
        elif os.path.isdir(cl_files_or_dir):
            all_files = glob.glob(os.path.join(cl_files_or_dir, '*.cl'))
            all_files += glob.glob(os.path.join(cl_files_or_dir, '.CL'))


        if isinstance(preferred_device, basestring):
            # should be in the same format that Theano uses to designate devices eg 'opencln:n'
            # or it should in the form of platform_name:device_name
