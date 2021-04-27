import numpy as np
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import pickle
import timeit
from CNN_Features import CNN_Features

class Data_Process(object):
	def __init__(self, pra_config):
		if not os.path.exists(pra_config.processed_data_root_path):
			os.makedirs(pra_config.processed_data_root_path)
		
		self.config = pra_config
		self.extract_cnn_features()
	
	def extract_cnn_features(self):
		cnn_feature_path = self.config.cnn_feature_path
		if os.path.exists(cnn_feature_path):
			with open(cnn_feature_path, 'r') as reader:
				self.cnn_feature_dict = pickle.load(reader)
				print 'CNN Featurs have been loaded.'
		else:
			image_folder_path = self.config.dataset_root_path
			num_before_frame = self.config.num_before_key_frame
			num_after_frame = self.config.num_after_key_frame
			result_image_path_list = []
			result_cnn_feature_list = []

			for v_folder in self.config.training + self.config.testing:
				v_annotations_path = os.path.join(image_folder_path, v_folder, 'annotations.txt')
				# print v_annotations_path
				with open(v_annotations_path, 'r') as reader:
					lines = [x.strip().split(' ') for x in reader.readlines()]
				
				lines = [x for x in lines if 'winpoint' not in x[1]]
				image_path_list = [os.path.join(image_folder_path, v_folder, x[0].split('.')[0], '{}.jpg'.format(y)) for x in lines for y in range(int(x[0].split('.')[0])-num_before_frame, int(x[0].split('.')[0])+num_after_frame+1)]
				result_image_path_list.extend(image_path_list)

			# Extract CNN Features
			result_cnn_feature_list = CNN_Features(self.config.cnn_model_path, result_image_path_list)
			
			# Build a path to CNN_feature dictionary
			self.cnn_feature_dict = {x:y for x,y in zip(result_image_path_list, result_cnn_feature_list)}

			print 'Saving CNN Features ...'
			with open(cnn_feature_path, 'w') as writer:
				pickle.dump(self.cnn_feature_dict, writer)
			print 'CNN Features have been saved in {}'.format(cnn_feature_path)


	# Build a sentence based on personal activity labels
	def build_sentence(self, pra_label, pra_action_set, pra_left, pra_right):
		left_sen = 'Left:'
		right_sen = 'Right:'
		sentence = '<SOS> {} {} {} {} {} <EOS>'.format(pra_label, left_sen, ' '.join(set(pra_left)), right_sen, ' '.join(set(pra_right)))
		return sentence


	def convert_vocab_to_index(self, pra_sentence_list):
		vocab_dictionary_path = self.config.caption_vocab_dictionary_path
		sentence_list = [x.split(' ') for x in pra_sentence_list]
		if os.path.exists(vocab_dictionary_path):
			with open(vocab_dictionary_path, 'r') as reader:
				vocab_dict = [x.strip() for x in reader.readlines()]
		else:
			vocab_dict = list(set(np.concatenate(sentence_list)))
			with open(vocab_dictionary_path, 'w') as writer:
				writer.write('\n'.join(vocab_dict))
		# set the the vocab size 
		self.config.caption_vocab_size = len(vocab_dict) 
		sentence_index = [[vocab_dict.index(x) for x in sent] for sent in sentence_list]
		return sentence_index


	# Get subset of data (return: image_path, cnn, caption, caption_index [label])
	def get_data_for_caption(self, pra_process='training'):
		image_folder_path = self.config.dataset_root_path
		result_image_path_list = []
		result_image_cnn_list = []
		result_image_caption_list = []
		result_image_label_list = []
		for v_folder in getattr(self.config, pra_process): #self.config.[pra_process]:
			v_annotations_path = os.path.join(image_folder_path, v_folder, 'annotations.txt')
			# print v_annotations_path
			with open(v_annotations_path, 'r') as reader:
				lines = [x.strip().split(' ') for x in reader.readlines()]
			
			lines = [x for x in lines if 'winpoint' not in x[1]]

			# Get image path
			image_path_list = [os.path.join(image_folder_path, v_folder, x[0].split('.')[0], x[0]) for x in lines]
			result_image_path_list.extend(image_path_list)

			# Get image CNN feature
			image_cnn_list = [self.cnn_feature_dict[x] for x in image_path_list]
			result_image_cnn_list.extend(image_cnn_list)

			# Get image activity label
			image_label_list = [x[1] for x in lines]
			
			# Get image caption
			player_action = [now_frame[6::5] for now_frame in lines]
			player_location = [[int(x) for x in now_frame[2::5]] for now_frame in lines]
			player_location_index = [np.argsort(x) for x in player_location]
			middle_number = [len(x)/2 for x in player_location]
			left = [[player_act[x] for x in location_index[:middle_num]] for player_act, location_index, middle_num in zip(player_action, player_location_index, middle_number)]
			right = [[player_act[x] for x in location_index[middle_num:]] for player_act, location_index, middle_num in zip(player_action, player_location_index, middle_number)]
			action_set = [set(x) for x in player_action]
			image_caption_list = [self.build_sentence(label, s,l,r) for label, s,l,r in zip(image_label_list, action_set, left, right)]
			# image_caption_list = [self.build_label_sentence(x) for x in image_label_list]
			result_image_caption_list.extend(image_caption_list)

		result_caption_index_list = self.convert_vocab_to_index(result_image_caption_list)
		return result_image_path_list, result_image_cnn_list, result_caption_index_list

	def convert_activity_to_index(self, pra_activity_list):
		activity_list_path = self.config.activity_list_path
		if os.path.exists(activity_list_path):
			with open(activity_list_path, 'r') as reader:
				activity_dict = [x.strip() for x in reader.readlines()]
		else:
			activity_dict = list(set(pra_activity_list))
			with open(activity_list_path, 'w') as writer:
				writer.write('\n'.join(activity_dict))
		
		activity_index = [activity_dict.index(x) for x in pra_activity_list]
		return activity_index
		

	# Get subset of data (return: image_path, cnn, label)
	def get_data_for_activity(self, pra_process='training'):
		num_before_frame = self.config.num_before_key_frame
		num_after_frame = self.config.num_after_key_frame
		image_folder_path = self.config.dataset_root_path
		result_image_path_list = []
		result_image_cnn_list = []
		result_image_label_list = []
		
		for v_folder in getattr(self.config, pra_process): #self.config.[pra_process]:
			v_annotations_path = os.path.join(image_folder_path, v_folder, 'annotations.txt')
			print v_annotations_path
			with open(v_annotations_path, 'r') as reader:
				lines = [x.strip().split(' ') for x in reader.readlines()]

			lines = [x for x in lines if 'winpoint' not in x[1]]
			
			# Get image path
			image_path_list = [os.path.join(image_folder_path, v_folder, x[0].split('.')[0], '{}.jpg'.format(y)) for x in lines for y in range(int(x[0].split('.')[0])-num_before_frame, int(x[0].split('.')[0])+num_after_frame+1)]
			result_image_path_list.extend(image_path_list)

			# Get image CNN feature
			image_cnn_list = [self.cnn_feature_dict[x] for x in image_path_list]
			result_image_cnn_list.extend(image_cnn_list)

			# Get image activity label
			image_label_list = [x[1] for x in lines]
			result_image_label_list.extend(image_label_list)

		result_image_label_list = self.convert_activity_to_index(result_image_label_list)
		return result_image_path_list, result_image_cnn_list, result_image_label_list

