import os
import numpy as np
import pickle
import tensorflow as tf 
from Configuration import FLAGS
from Data_Process import Data_Process
from Caption_Model import Caption_Model
# from Activity_Model import Activity_Model

def train_models():
	data_process = Data_Process(FLAGS)
	
	# 1. Data process for lstm_caption
	print 'Getting trianing data for Caption_Model ...'
	image_path_list, image_cnn_list, image_caption_index_list = data_process.get_data_for_caption('training')
	
	# 2. Train lstm_caption model
	print 'Trianing Caption_Model ...'
	caption_model = Caption_Model(FLAGS)
	caption_model.train_model(image_path_list, image_cnn_list, image_caption_index_list)

	# 3. Data process for lstm_activity
	print 'Generating Captions for Activity_Model to train ...'
	act_train_image_path, act_train_image_cnn, act_train_image_label = data_process.get_data_for_activity('training')
	predicted_caption = caption_model.test_model(act_train_image_path, act_train_image_cnn)
	

def main(_):
	train_models()


if __name__ == '__main__':
	tf.app.run()