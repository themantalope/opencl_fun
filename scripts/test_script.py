
# coding: utf-8

# # First pyopencl notebook and program
# [Based on the code from this blog post.](https://karthikhegde.blogspot.com/2013/09/hope-you-liked-previous-introductory.html)

# In[4]:

import os
import pyopencl as pcl
import numpy as np


# In[54]:

# let's try to find available devices
platforms = pcl.get_platforms()
for p in platforms:
    devs = p.get_devices()
    for d in devs:
        print(d.name, pcl.device_type.to_string(d.type), d.global_mem_size / 10**9)


# In[55]:

# let's select the AMD radeon card in this case
dev=None
for p in pcl.get_platforms():
    devs = p.get_devices()
    for d in devs:
        if pcl.device_type.to_string(d.type) == 'GPU' and (d.global_mem_size / 10**9) > 2.0:
            dev = d


# In[57]:

print(dev.name)


# In[74]:

# make the opencl context
# cntx = pcl.create_some_context()
cntx = pcl.Context(devices=[dev])
queue = pcl.CommandQueue(cntx, device=dev)


# In[75]:

# make the numpy arrays
n1 = np.arange(0, 10, dtype=np.int32)
n2 = np.arange(0, 10, dtype=np.int32)
out = np.empty(shape = n1.shape, dtype=np.int32)


# In[76]:

# create the opencl buffers
n1_buf = pcl.Buffer(cntx, pcl.mem_flags.READ_ONLY | pcl.mem_flags.COPY_HOST_PTR, hostbuf=n1)
n2_buf = pcl.Buffer(cntx, pcl.mem_flags.READ_ONLY | pcl.mem_flags.COPY_HOST_PTR, hostbuf=n2)
out_buf = pcl.Buffer(cntx, pcl.mem_flags.WRITE_ONLY, out.nbytes)


# In[77]:

# get the location of the opencl code
first_cl_file = os.path.join('..', 'src', 'cl', 'first.cl')
os.path.isfile(first_cl_file)


# In[78]:

# build the kernel
with open(first_cl_file, 'r') as f:
    build = pcl.Program(cntx, f.read()).build()
    


# In[79]:

# launch the kernel
launch = build.first(queue, n1.shape, n2.shape, n1_buf, n2_buf, out_buf)


# In[80]:

launch.wait()


# In[81]:

# read the output
pcl.enqueue_copy(queue, out, out_buf).wait()


# In[82]:

print(out)


# In[87]:

# save this notebook to a script
get_ipython().system("jupyter nbconvert --to script test_notebook.ipynb --output='../scripts/test_script'")


# In[86]:

# !jupyter nbconvert --help


# In[ ]:



