def lstm_model_2(num_features = 1280, seq_length = 7, num_classes = 3):

	net = Sequential()
	net.add(LSTM(num_features,
	return_sequences=False,
	input_shape=(seq_length, num_features),
	dropout=0.5))
	net.add(Dense(512, activation='relu'))
	net.add(Dropout(0.5))
	net.add(Dense(num_classes, activation='softmax'))

	return net

def convolution_3d(num_features = 1280, seq_length = 7, num_classes = 3):
	net = Sequential()
	net.add(Conv3D(32,
	kernel_size=(3, 3, 3),
	input_shape=(seq_length,num_features)))
	net.add(Activation('relu'))
	net.add(Conv3D(32, (3, 3, 3)))
	net.add(Activation('softmax'))
	net.add(MaxPooling3D())
	net.add(Dropout(0.25))
	net.add(Conv3D(64, (3, 3, 3)))
	net.add(Activation('relu'))
	net.add(Conv3D(64, (3, 3, 3)))
	net.add(Activation('softmax'))
	net.add(MaxPool3D())
	net.add(Dropout(0.25))
	net.add(Flatten())
	net.add(Dense(512, activation='sigmoid'))
	net.add(Dropout(0.5))
	net.add(Dense(num_classes, activation='softmax'))
	
	return net

def lstm_model_3(num_features = 1280, seq_length = 7, hidden_units = 256, dense_units = 256, reg = 1e-1, dropout_rate=1e-1, num_classes = 3):

	model = Sequential()
	model.add(Dropout(dropout_rate, input_shape = (seq_length, num_features)))
	model.add(Bidirectional(LSTM(hidden_units, return_sequences = True)))
	model.add(TimeDistributed(Dropout(dropout_rate)))
	model.add(TimeDistributed(Dense(num_classes, activation = 'softmax')))
	
	averaging_layer = Lambda(function = lambda  x : K.mean(x,axis = 1))

	model.add(averaging_layer)

	return model