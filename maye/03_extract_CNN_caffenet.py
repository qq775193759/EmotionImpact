import numpy as np
import Image
import glob
import matplotlib
matplotlib.use('tkagg')		# for X11 forwarding

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/media/hcsi3/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import os

caffe.set_mode_gpu()
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

#image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
# transformed_image = transformer.preprocess('data', image)

d = '/media/hcsi3/EmotionImpact/2016data/frames/'
NN = 9800
all_out = []
for i in range(NN):
	dd = d + str(i) + '/'
	g = glob.glob(dd + '*.png')
	data = []
	net.blobs['data'].reshape(len(g), 3, 227, 227)  # image size is 227x227
	for s in g:
		img = Image.open(s)
		img = img.resize((256, 256), Image.ANTIALIAS)
		width, height = img.size
		left = (width - 227)/2
		top = (height - 227)/2
		right = (width + 227)/2
		bottom = (height + 227)/2
		img = img.crop((left, top, right, bottom))
		image = np.array(img)
		transformed_image = transformer.preprocess('data', image)
		data.append(transformed_image)
	net.blobs['data'].data[...] = np.array(data)
	output = net.forward()
	output = net.blobs['fc7'].data
	# Do average for simplicity
	out = np.mean(output, axis=0)
	all_out.append(out)

with open('caffeNet_4096_avg.txt', 'w') as f:
	for i in range(NN):
		f.write(str(i))
		for j in all_out[i]:
			f.write('\t' + str(j))
		f.write('\n')
