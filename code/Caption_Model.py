import tensorflow as tf 
import numpy as np 
import os
import random 

class LSTMRNN(object):
	def __init__(self, pra_config):
		self.batch_size = pra_config.caption_batch_size
		self.image_dimension = pra_config.caption_image_dimension
		self.vocab_size = pra_config.caption_vocab_size
		self.embed_size = pra_config.caption_embed_size
		self.hidden_dim = pra_config.caption_hidden_dim
		self.lstm_layers = pra_config.caption_lstm_layers
		self.learning_rate = pra_config.caption_learning_rate

		with tf.variable_scope('Caption'):
			self.global_step = tf.Variable(
				initial_value = 0,
				name = 'global_step',
				trainable = False#,
				# collections = [tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES]
				)

			# Data input
			self.image_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.image_dimension], 'image_cnn')
			self.sentence_placeholder = tf.placeholder(tf.int32, [self.batch_size, None], 'sentence')
			self.target_placehoder = tf.placeholder(tf.int32, [self.batch_size, None], 'target')
			self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')
		
		# Image Input Layer
		with tf.variable_scope('Image_Input_Layer'):
			w = self.get_weight([self.image_dimension, self.embed_size])
			b = self.get_bias([self.embed_size,])
			wx_plus_b = tf.matmul(self.image_placeholder, w) + b
			self.image_cnn = tf.expand_dims(wx_plus_b, 1)
		# Sentence Input Layer
		with tf.variable_scope('Sentence_Input_Layer'):
			w = self.get_weight([self.vocab_size, self.embed_size])
			self.sent_embedup = tf.nn.embedding_lookup(w, self.sentence_placeholder)
		# All Input Layer
		with tf.variable_scope('All_Input_Layer'):
			self.all_input = tf.concat(1, [self.image_cnn, self.sent_embedup])
		# LSTM Layer
		with tf.variable_scope('LSTM_Layer'):
			lstm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, forget_bias=1, input_size=self.embed_size)
			lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)
			stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_dropout]*self.lstm_layers)
			self.lstm_init = stacked_lstm.zero_state(self.batch_size, tf.float32)
			self.lstm_output, self.state = tf.nn.dynamic_rnn(stacked_lstm, self.all_input, initial_state=self.lstm_init)
		# Prediction
		with tf.variable_scope('Prediction'):
			reshape_lstm_output = tf.reshape(self.lstm_output, [-1, self.hidden_dim])
			w = self.get_weight([self.hidden_dim, self.vocab_size])
			b = self.get_bias([self.vocab_size,])
			self.prediction = tf.matmul(reshape_lstm_output, w) + b
			self.predicted_label = tf.argmax(self.prediction, 1)
		# Loss
		with tf.variable_scope('Loss'):
			sentence_target = tf.reshape(self.target_placehoder, [-1])
			self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.prediction, sentence_target, name='loss'))
		# Train
		with tf.variable_scope('Train'):
			self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)

	def get_weight(self, pra_shape, pra_name='Weight'):
		init = tf.random_normal_initializer(mean=0.0, stddev=1.)
		return tf.get_variable(name=pra_name, shape=pra_shape, initializer=init)

	def get_bias(self, pra_shape, pra_name='Bias'):
		init = tf.constant_initializer(0.1)
		return tf.get_variable(name=pra_name, shape=pra_shape, initializer=init)



class Caption_Model(object):
	def __init__(self, pra_config):#, pra_image_path_list, pra_image_cnn_list, pra_image_caption_index_list):
		self.config = pra_config
		vocab_dictionary_path = self.config.caption_vocab_dictionary_path
		if os.path.exists(vocab_dictionary_path):
			with open(vocab_dictionary_path, 'r') as reader:
				self.vocab_dict = [x.strip() for x in reader.readlines()]

		if not os.path.exists(pra_config.caption_model_folder):
			os.makedirs(pra_config.caption_model_folder)

		# self.image_path_list = pra_image_path_list
		# self.image_cnn_list = pra_image_cnn_list
		# self.image_caption_index_list = pra_image_caption_index_list

		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph)
		with self.sess.graph.as_default():
			self.lstm = LSTMRNN(pra_config)

	def get_next_training_batch(self):
		for i in xrange(self.config.caption_iteration):
			image_path_list = self.image_path_list
			image_cnn_list = self.image_cnn_list
			image_caption_index_list = self.image_caption_index_list
			
			index = random.sample(range(len(image_path_list)), self.config.caption_batch_size)

			sentence = np.array([image_caption_index_list[x] for x in index])
			cnn_feature = [image_cnn_list[x] for x in index]
			sentence_target = sentence[:, 1:]
			added_dim = [[0]]*self.config.caption_batch_size
			sentence_target = np.array([list(y) + list(x) + list(y) for x,y in zip(sentence_target, added_dim)])
			yield sentence, cnn_feature, sentence_target

	def train_model(self, pra_image_path_list, pra_image_cnn_list, pra_image_caption_index_list):
		self.image_path_list = pra_image_path_list
		self.image_cnn_list = pra_image_cnn_list
		self.image_caption_index_list = pra_image_caption_index_list

		with self.sess.graph.as_default(), tf.Session() as sess:
			writer = tf.train.SummaryWriter(self.config.caption_summary_folder, sess.graph)
			model_saver = tf.train.Saver()
			# init = tf.initialize_all_variables()
			init = tf.global_variables_initializer()
			sess_manager = tf.train.SessionManager()
			sess = sess_manager.prepare_session("", init_op=init, saver=model_saver, checkpoint_dir=self.config.caption_model_folder)
			for sentence, image, target in self.get_next_training_batch():
				feed = {
					self.lstm.sentence_placeholder: sentence,
					self.lstm.image_placeholder: image,
					self.lstm.target_placehoder: target,
					self.lstm.dropout_placeholder: self.config.caption_dropout
				}
				_, loss, predicted, global_step = sess.run([self.lstm.train_op, self.lstm.loss, self.lstm.predicted_label, self.lstm.global_step], feed_dict=feed)
				w_sent = ' '.join([self.vocab_dict[int(x)] for x in sentence[0]])
				w_pred = ' '.join([self.vocab_dict[x] for x in predicted[1:-1]])
				print 'Iteration:{}, Loss:{:0.3f}, \nSentence: {}, \nPredicted: {}\n'.format(global_step, loss, w_sent, w_pred)
				if global_step%self.config.caption_save_per_time==0:
					model_saver.save(sess, self.config.caption_model_save_path, global_step=global_step)
				if global_step>=self.config.caption_iteration:
					break

	# def get_sentence_descriptor(self, pra_image_path, pra_image_cnn):
	# 	self.sess.graph.as_default()
	# 	with tf.Session() as sess:
	# 	# with self.sess.graph.as_default(), tf.Session() as sess:
	# 		model_saver = tf.train.Saver()
	#		init = tf.initialize_all_variables()
	#		sess.run(init)
	#		init = tf.global_variables_initializer()
	# 		sess_manager = tf.train.SessionManager()
	# 		sess = sess_manager.prepare_session("", init_op=init, saver=model_saver, checkpoint_dir=self.config.caption_model_folder)
	# 		result_descriptor = []
	# 		for step, (image_path, image_cnn) in enumerate(zip(pra_image_path, pra_image_cnn)):
	# 			sent = np.ones([self.config.caption_batch_size, 1]) * self.vocab_dict.index('<SOS>')
	# 			while sent[0][-1]!=self.vocab_dict.index('<EOS>') and len(sent[0])<150:
	# 				feed = {
	# 					self.lstm.sentence_placeholder: sent,
	# 					self.lstm.image_placeholder: [image_cnn],
	# 					# self.lstm.target_placehoder: target,
	# 					self.lstm.dropout_placeholder: 1
	# 				}
	# 				predicted = sess.run([self.lstm.predicted_label], feed_dict=feed)
	# 				new_word = np.array(predicted)[:,-1:]
	# 				sent = np.concatenate([sent, new_word], 1)
	# 			result_descriptor.append(predicted[0][1:-1])
	# 		return result_descriptor

	def test_model(self, pra_image_path, pra_image_cnn):
		with self.sess.graph.as_default(), tf.Session() as sess:
			model_saver = tf.train.Saver()
			# init = tf.initialize_all_variables()
			init = tf.global_variables_initializer()
			sess_manager = tf.train.SessionManager()
			sess = sess_manager.prepare_session("", init_op=init, saver=model_saver, checkpoint_dir=self.config.caption_model_folder)
			result_descriptor = []
			for step, (image_path, image_cnn) in enumerate(zip(pra_image_path, pra_image_cnn)):
				sent = np.ones([self.config.caption_batch_size, 1]) * self.vocab_dict.index('<SOS>')
				while sent[0][-1]!=self.vocab_dict.index('<EOS>') and len(sent[0])<self.config.caption_max_length:
					feed = {
						self.lstm.sentence_placeholder: sent,
						self.lstm.image_placeholder: [image_cnn],
						# self.lstm.target_placehoder: target,
						self.lstm.dropout_placeholder: 1
					}
					predicted = sess.run([self.lstm.predicted_label], feed_dict=feed)
					new_word = np.array(predicted)[:,-1:]
					sent = np.concatenate([sent, new_word], 1)
				# result_descriptor.append(predicted[0][1:-1])
				w_pred = ' '.join([self.vocab_dict[x] for x in predicted[0][1:-1]])
				result_descriptor.append(w_pred)
				print 'Test Model:{}, Image:{:50}, Predicted:{:50}'.format(step, image_path, w_pred)
				# writer.write('{}\t{}\n'.format(image_path, w_pred))
			return result_descriptor

if __name__ == '__main__':
	pass
	# train_caption_model()
	# test_model()

