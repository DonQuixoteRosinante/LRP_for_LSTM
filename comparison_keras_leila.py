from keras.layers import LSTM, Dense, Bidirectional
from keras.models import Sequential
import pickle as pkl
from code.LSTM.LSTM_bidi import *
import codecs


def get_test_sentence(sent_idx):
    """Returns a test set sentence and its label, sent_idx must be an integer in [1, 2210]"""
    idx = 1
    with codecs.open("./data/sequence_test.txt", 'r', encoding='utf8') as f:
        for line in f:
            line          = line.rstrip('\n')
            line          = line.split('\t')
            label         = int(line[0])-1         # true sentence class
            words         = line[1].split(' | ')   # sentence words
            if idx == sent_idx:
                return words, label
            idx +=1


# the input,forget, cell and output gate are all saved in one matrix. Order between leilas implementation and keras
# differs, i.e. second and third quarter of the matrix have to be switched and the matrix has to be transposed
def keras_matrix_to_leila_matrix(keras_weights):
    no_units = int(keras_weights.shape[1]/4)
    sorted_weight_matrix = np.zeros(shape=keras_weights.shape)
    # input gate
    sorted_weight_matrix[:, :no_units] = keras_weights[:, :no_units]
    # cell gate / g gate
    sorted_weight_matrix[:, no_units:2*no_units] = keras_weights[:, 2*no_units:3*no_units]
    # forget gate
    sorted_weight_matrix[:, 2*no_units:3*no_units] = keras_weights[:, no_units:2*no_units]
    # output gate
    sorted_weight_matrix[:, 3*no_units:] = keras_weights[:, 3*no_units:]
    return np.transpose(sorted_weight_matrix)


# bias vectors are in the wrong order aswell so we swap the 2nd and 3rd quarter
def keras_bias_to_leila_bias(keras_bias):
    no_units = int(len(keras_bias)/4)
    sorted_bias = np.zeros(shape=keras_bias.shape)
    # input gate
    sorted_bias[:no_units] = keras_bias[:no_units]
    # cell gate / g gate
    sorted_bias[no_units:2*no_units] = keras_bias[2*no_units:3*no_units]
    # forget gate
    sorted_bias[2*no_units:3*no_units] = keras_bias[no_units:2*no_units]
    # output gate
    sorted_bias[3*no_units:] = keras_bias[3*no_units:]
    return sorted_bias


# convert keras model to a dict suitable for leila lstm model representation
def keras_model_to_leila_dict(model):
    leila_dict = {}
    keras_weights = model.get_weights()
    # sort the keras weight matrix for recurrent units to fit into leila format
    leila_dict['Wxh_Left'] = keras_matrix_to_leila_matrix(keras_weights[0])
    leila_dict['Whh_Left'] = keras_matrix_to_leila_matrix(keras_weights[1])
    leila_dict['Wxh_Right'] = keras_matrix_to_leila_matrix(keras_weights[3])
    leila_dict['Whh_Right'] = keras_matrix_to_leila_matrix(keras_weights[4])
    # the biases of the recurrent units have to be split to fit into the two bias approach
    leila_dict['bxh_Left'] = 0.5 * keras_bias_to_leila_bias(keras_weights[2])
    leila_dict['bhh_Left'] = 0.5 * keras_bias_to_leila_bias(keras_weights[2])
    leila_dict['bxh_Right'] = 0.5 * keras_bias_to_leila_bias(keras_weights[5])
    leila_dict['bhh_Right'] = 0.5 * keras_bias_to_leila_bias(keras_weights[5])
    # finally the hidden layer matrix just has to be transposed
    no_features = keras_weights[6].shape[0]
    leila_dict['Why_Left'] = np.transpose(keras_weights[6][:int(no_features/2), :])
    leila_dict['Why_Right'] =np.transpose(keras_weights[6][int(no_features/2):, :])
    return leila_dict


batch_size = 32
seq_len = 40
feature_size = 60
units = 60


words, target_class = get_test_sentence(291)
data = np.random.randn(batch_size, seq_len, feature_size)
# print(words)
# print(target_class)

model = Sequential()
model.add(Bidirectional(LSTM(units=units, activation='tanh', recurrent_activation='sigmoid'),
                        input_shape=(None, feature_size), merge_mode='concat'))
model.add(Dense(5, use_bias=False))
model.load_weights('model_comparison/random_weights')


net = LSTM_bidi(model_path='model_comparison/')
w_indices = [net.voc.index(w) for w in words]
#print(w_indices)
T      = len(w_indices)                         # input word sequence length
e      = net.E.shape[1]                # word embedding dimension
x      = np.zeros((T, e))
x[:, :] = net.E[w_indices, :]


net.set_input(w_indices)
leila_output = net.forward()
print('input_shape:', x.shape)
print('leila output shape:', leila_output.shape)
#print('leila hidden state vector left', net.h_Left)
#print('leila hidden state vector right', net.h_Right)

y = model.predict(x.reshape((1,T,e)))
#model.save_weights('model_comparison/random_weights')
#y = model.predict(data)

for i in range(len(model.layers)):
    for e in zip(model.layers[i].trainable_weights, model.layers[i].get_weights()):
        print('Param %s:\n%s' % (e[0], e[1].shape))

model.summary()
# converted_weights = keras_model_to_leila_dict(model)
# pkl.dump(converted_weights, open('model_comparison/model', 'wb'))
print('keras_model_output', y)
print('leila model output', leila_output)

eps = 0.001
bias_factor = 0.0  # recommended value

Rx, Rx_rev, R_rest = net.lrp(w_indices, target_class, eps, bias_factor)  # LRP through the net
R_words = np.sum(Rx + Rx_rev, axis=1)  # word relevances

scores = net.s.copy()  # classification
print("prediction scores:", scores)
print("\nLRP target class:         ", target_class)
print("\nLRP relevances:")
for idx, w in enumerate(words):
    print("\t\t\t" + "{:8.2f}".format(R_words[idx]) + "\t" + w)