import os
import tensorflow as tf 

FLAGS = tf.app.flags.FLAGS

# Configuration of dataset
tf.app.flags.DEFINE_string('dataset_root_path', '/Users/xin/Documents/Dataset/volleyball/',
								'The path of image database')
tf.app.flags.DEFINE_string('training', ['1', '3', '6', '7', '10', '13', '15', '16', '18', '22', '23', '31', '32', '36', '38', '39', '40', '41', '42', '48', '50', '52', '53', '54', '0', '2', '8', '12', '17', '19', '24', '26', '27', '28', '30', '33', '46', '49', '51'],
								'Training subset of Volleyball Database')
tf.app.flags.DEFINE_string('testing', ['4', '5', '9', '11', '14', '20', '21', '25', '29', '34', '35', '37', '43', '44', '45', '47'],
								'Testing subset of Volleyball Database')

tf.app.flags.DEFINE_string('cnn_model_path', '../models/classify_image_graph_def.pb',
								'The path of CNN model')

# Configuration for cache path
tf.app.flags.DEFINE_string('cache_root_path', '../data_cache',
								'The root path of cache data')
tf.app.flags.DEFINE_string('processed_data_root_path', os.path.join(FLAGS.cache_root_path, 'data'),
								'Pre-processed data will be stored in this path')
tf.app.flags.DEFINE_string('cnn_feature_path', os.path.join(FLAGS.cache_root_path, 'data/CNN_Feature.pkl'),
						  		'Extracted CNN features will be stored in this path')



# #################################################################################
# ################## Configuration for Caption and Activity #######################
# 
# ############################### Modify Manually #################################
tf.app.flags.DEFINE_integer('num_before_key_frame', 5, 
								'The number of frames before the current frame.')
tf.app.flags.DEFINE_integer('num_after_key_frame', 4, 
								'The number of frames after the current frame.')
tf.app.flags.DEFINE_integer('caption_max_length', 20,
								'The max length of caption')
# #################################################################################




# #################################################################################
# ########################## Configuration for Caption ############################
# 
# ############################# Modify Manually ###################################
tf.app.flags.DEFINE_integer('caption_iteration', 100000,
								'Iteration times during training Caption LSTM Model.')
tf.app.flags.DEFINE_integer('caption_save_per_time', 1000,
								'Save model weight per K(e.g. K=1000) times')
tf.app.flags.DEFINE_string('caption_model_folder', os.path.join(FLAGS.cache_root_path, 'caption_model'),
								'Caption Model will be saved in this path')
tf.app.flags.DEFINE_string('caption_model_save_path', os.path.join(FLAGS.cache_root_path, 'caption_model/my_caption_model'),
								'Caption Model name')
tf.app.flags.DEFINE_string('caption_summary_folder', os.path.join(FLAGS.cache_root_path, 'caption_log'),
								'Caption Model log will be saved in this path')
tf.app.flags.DEFINE_string('caption_vocab_dictionary_path', os.path.join(FLAGS.cache_root_path, 'data/caption_vocab_dictionary.txt'),
								'Vocab dictionary will be stored in this path')
# #################### Update in runtime automatically ############################
tf.app.flags.DEFINE_integer('caption_vocab_size', 0,
								'The number of unique Vocab used in image caption') 
# ################# Static parameters (Do not need to modify them) ################
tf.app.flags.DEFINE_integer('caption_batch_size', 1,
								'Batch size during training Caption LSTM Model.')
tf.app.flags.DEFINE_integer('caption_image_dimension', 2048,
								'Dimension of Image CNN Feature')
tf.app.flags.DEFINE_integer('caption_embed_size', 1024,
								'Embed size during training Caption LSTM Model.')
tf.app.flags.DEFINE_integer('caption_hidden_dim', 1024,
								'Hidden dimension during training Caption LSTM Model.')
tf.app.flags.DEFINE_integer('caption_lstm_layers', 2,
								'Number of layers in training Caption LSTM Model.')
tf.app.flags.DEFINE_float('caption_learning_rate', 1e-4,
								'Learning rate during training Caption LSTM Model.')
tf.app.flags.DEFINE_float('caption_dropout', 0.75,
								'Dropout during training Caption LSTM Model.')
# #################################################################################



