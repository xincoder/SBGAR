import tensorflow as tf
from tensorflow.python.platform import gfile
import timeit
import numpy as np

def CNN_Features(pra_cnn_model_path, pra_image_paths, verbose=True):
	"""
	extract_cnn_features computed the inception bottleneck feature for a list of images

	image_paths: array of image path
	return: 2-d array in the shape of (len(image_paths), 2048)
	"""
	# print image_paths	
	with gfile.FastGFile(pra_cnn_model_path, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

	feature_dimension = 2048
	features = np.empty((len(pra_image_paths), feature_dimension))

	with tf.Session() as sess:
		flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')

		time_list = []
		for i, image_path in enumerate(pra_image_paths):
			if verbose:
				print('Processing %s...' % (image_path))
			if not gfile.Exists(image_path):
				tf.logging.fatal('File does not exist %s', image)

			time_start = timeit.default_timer()
			image_data = gfile.FastGFile(image_path, 'rb').read()
			feature = sess.run(flattened_tensor, {'DecodeJpeg/contents:0': image_data})
			features[i, :] = np.squeeze(feature)
			time_end = timeit.default_timer()
			time_list.append(time_end - time_start)
		print 'Extract CNN Feature time (average):', np.mean(time_list)
	return features 