from __future__ import division
import numpy as np
import csv
import scipy.io.wavfile
import sys
from Afilter import A_weighting
from scipy.signal import lfilter


def read_from_disk(data_dir, data_list):
    """Reads txt file containing paths to sound files and ground truth labels.
    Loads all the data into memory, after basic preprocessing

    Args:
      data_dir: path to the directory of all the audio files
      data_list: path to the csv file with the data description

    Returns:
      Lists with all file names, labels, and is_esc10 (esc10 is a smaller dataset).
    """
    with open(data_list) as csvfile:
        reader = csv.DictReader(csvfile)
        labels = []
        is_esc10 = []
        count = 0
        out_array = np.zeros((2000, 220500), dtype=np.int16)
        for row in reader:
            labels.append(int(row['target']))
            is_esc10.append(row['esc10'])
            [_, data] = scipy.io.wavfile.read(data_dir + row['filename'])
            out_array[count, :] = data
            count = count + 1

    print(sys.getsizeof(out_array))
    return out_array , np.asarray(labels), is_esc10

def preprocess_base_data_batch(data, required_size, isTrain):
    """recieves batch of inputs, and does the following preprocess steps for every signal in the batch
    1. trim zeros for start/end of the file
    2. zero padding on both sides, with required_size/2 zeros
    3. cropping input

    Args:
      data: batch of train inputs
      required_size: required output size. output dimensions will be: [batch,required_size]

    Returns:
      preprocessed data, shape = [batch_size,required_size]
    """
    out_array = np.zeros((data.shape[0], required_size), dtype=np.float)
    for m in range(data.shape[0]):
        signal = preprocess_base_train_data_for_1_signal(data[m, :], required_size, isTrain)
        out_array[m,:] = signal
    return out_array


def preprocess_base_train_data_for_1_signal(signal, required_size,isRandom):
    """receives 1 inputs signal, and does the following preprocess steps:
    1. trim zeros for start/end of the file
    2. zero padding on both sides, with required_size/2 zeros
    3. cropping input according to crop_index and required_size
    4. isRandom is true for training, false for validation

    Args:
      data: 1 signal
      required_size: required output size. output dimensions will be: [batch,required_size]

    Returns:
      preprocessed data
    """
    # trim zeros
    trimmed_signal = np.trim_zeros(signal)

    # zero padding
    zero_padding = int(required_size / 2 + 1)
    trimmed_signal_zero_padded = np.pad(trimmed_signal, (zero_padding, zero_padding), 'constant', constant_values=(0, 0))

    # cropping
    #index = np.random.randint(0, high=(trimmed_signal_zero_padded.shape[0] - required_size))
    if isRandom == True:
        index = np.random.randint(0, high=(trimmed_signal_zero_padded.shape[0] - required_size))
    else:
        if trimmed_signal_zero_padded.shape[0] < 2*required_size:
            index = 0
        else:
            index = zero_padding

    out_signal = trimmed_signal_zero_padded[index:index + required_size]

    return out_signal

def preprocess_base_valid_data_for_1_signal(self, signal, input_size, num_valid_section ):
        """receives 1 inputs signal, and does the following preprocess steps:
        1. trim zeros for start/end of the file
        2. zero padding on both sides, with required_size/2 zeros
        3. crops num_valid_section subarrays, each one with size of input_size
        Args:
          data: 1 signal
        Returns:
          preprocessed data
        """
        # trim zeros
        trimmed_signal = np.trim_zeros(signal)

        # zero padding
        zero_padding = int(input_size / 2 + 1)
        trimmed_signal_zero_padded = np.pad(trimmed_signal, (zero_padding, zero_padding), 'constant',
                                            constant_values=(0, 0))

        # cropping indecies
        L = trimmed_signal_zero_padded.size
        valid_stride = int((L - input_size) / (num_valid_section - 1))
        crop_indecies = range(0, L - input_size + 1, valid_stride)

        out_array = np.zeros((num_valid_section, input_size), dtype=np.float)
        for m in range(num_valid_section):
            out_array[m, :] = trimmed_signal_zero_padded[crop_indecies[m]:crop_indecies[m]+input_size]

        return out_array


def preprocess_base_valid_data_for_fold(signal, input_size, num_valid_section):
    """receives 1 inputs signal, and does the following preprocess steps:
    1. trim zeros for start/end of the file
    2. zero padding on both sides, with required_size/2 zeros
    3. crops num_valid_section subarrays, each one with size of input_size
    Args:
      data: 1 signal
    Returns:
      preprocessed data
    """
    # trim zeros
    trimmed_signal = np.trim_zeros(signal)

    # zero padding
    zero_padding = int(input_size / 2 + 1)
    trimmed_signal_zero_padded = np.pad(trimmed_signal, (zero_padding, zero_padding), 'constant',
                                        constant_values=(0, 0))

    # cropping indecies
    L = trimmed_signal_zero_padded.size
    valid_stride = int((L - input_size) / (num_valid_section - 1))
    crop_indecies = range(0, L - input_size + 1, valid_stride)

    out_array = np.zeros((num_valid_section, input_size), dtype=np.float)
    for m in range(num_valid_section):
        out_array[m, :] = trimmed_signal_zero_padded[crop_indecies[m]:crop_indecies[m] + input_size]

    return out_array







class SoundReaderKCrossValidation(object):
    '''Generic SoundReader which reads sound files and their labels.
    It splits the data into k cross validation section (after the needed preprocessing
    '''

    def __init__(self, data_dir, data_list, k, requierd_input_size,number_of_classes, num_valid_section):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          k: number of cross validation section (typically 5)
        '''
        
        self.data_dir = data_dir
        self.data_list = data_list
        self.k = k
        self.requierd_input_size = requierd_input_size
        self.data, self.labels, self.is_ec10 = read_from_disk(self.data_dir,self.data_list )
        self.fold_length = int(len(self.data) / self.k)
        self.data_indecies = np.arange(len(self.data))
        self.num_valid_section = num_valid_section
        self.number_of_classes = number_of_classes

        #save trainin/cv indecies
        self.train_index_array = np.zeros((self.k,self.fold_length*(self.k-1)), dtype=np.int)
        self.cv_index_array = np.zeros((self.k, self.fold_length) , dtype=np.int)
        for m in range(self.k):
            lower_bound = self.data_indecies >= m * self.fold_length
            upper_bound = self.data_indecies < (m + 1) * self.fold_length
            cv_region = lower_bound * upper_bound
            self.train_index_array[m,:] = self.data_indecies[np.nonzero(~cv_region)]
            self.cv_index_array[m,:] = self.data_indecies[np.nonzero(cv_region)]



    def get_batch(self,fold_index,train_offset,batch_size):
        #start_index = self.train_index_array[fold_index,train_offset]
        #end_index = self.train_index_array[fold_index,train_offset+ batch_size-1]+1
        indices = self.train_index_array[fold_index,train_offset:train_offset+ batch_size]
        #print(indecies)
        #raw_data = np.take( self.data, indices,axis=0)
        raw_data = self.data[indices,:]
        raw_data = raw_data.astype(float)
        # normalization between -1 to 1
        raw_data = raw_data / (1 << 15)
        #print('fold {:d} \t offset = {:d},start - {:d},end - {:d}'.format(fold_index, train_offset, start_index,end_index))
        # one hot encoding
        indecies = np.arange(batch_size)
        one_hot_label = np.zeros((batch_size, self.number_of_classes))
        one_hot_label[indecies, self.labels[indices]] = 1
        return preprocess_base_data_batch(raw_data, self.requierd_input_size,True), one_hot_label

    def get_train_length(self):
        return self.train_index_array.shape[1]

    def get_valid_length(self):
        return self.cv_index_array.shape[1]

    def get_validation_samples_of_1_input(self, fold_index, offset):
        index = self.cv_index_array[fold_index, offset]
        raw_data = self.data[index, :]
        raw_data = raw_data.astype(float)
        # normalization between -1 to 1
        raw_data = raw_data / (1 << 15)

        out_mat = preprocess_base_valid_data_for_1_signal(self, raw_data, self.requierd_input_size,
                                                          self.num_valid_section)

        # one hot encoding
        indecies = np.arange(self.num_valid_section)
        one_hot_label = np.zeros((self.num_valid_section, self.number_of_classes))
        one_hot_label[indecies, self.labels[index]] = 1

        return out_mat, one_hot_label



    def get_validation_batch_10_crops(self, fold_index, offset, batch_size):
        out_data = np.zeros((batch_size*self.num_valid_section, self.requierd_input_size))
        one_hot_label = np.zeros((batch_size*self.num_valid_section, self.number_of_classes))

        for m in range(batch_size):
            data, label = self.get_validation_samples_of_1_input(fold_index, offset + m)
            out_data[m*self.num_valid_section:(m+1)*self.num_valid_section, :] = data
            one_hot_label[m*self.num_valid_section:(m+1)*self.num_valid_section, :] = label
        return out_data, one_hot_label


    def get_validation_batch(self, fold_index, offset, batch_size):
        start_index = self.cv_index_array[fold_index, offset]
        end_index = self.cv_index_array[fold_index, offset + batch_size-1] + 1
        raw_data = self.data[start_index:end_index, :]
        raw_data = raw_data.astype(float)
        # normalization between -1 to 1
        raw_data = raw_data / (1 << 15)
        # one hot encoding
        indecies = np.arange(batch_size)
        one_hot_label = np.zeros((batch_size, self.number_of_classes))
        one_hot_label[indecies, self.labels[start_index:end_index]] = 1

        return preprocess_base_data_batch(raw_data, self.requierd_input_size, True), one_hot_label

    def get_batch_bc(self, fold_index,train_offset, batch_size):
        indices = self.train_index_array[fold_index,train_offset:train_offset+ batch_size]

        raw_data = self.data[indices, :]
        raw_data = raw_data.astype(float)
        # normalization between -1 to 1
        raw_data = raw_data / (1 << 15)

        data_array = preprocess_base_data_batch(raw_data, self.requierd_input_size, True)

        labels = self.labels[indices]

        # Mix Data

        # get pairs of unique elements
        permuation1 = np.random.permutation(batch_size)
        permuation2 = np.random.permutation(batch_size)

        max_iteration = 20
        for m in range(max_iteration):
            indector1 = permuation1 == permuation2
            indector2 = labels[permuation1] == labels[permuation2]
            indector = indector1 + indector2
            indecies_not_unique = np.nonzero(indector)
            indecies_not_unique = np.asarray(indecies_not_unique)
            if indecies_not_unique.size == 0:
                break
            else:
                temp_permutation = np.random.permutation(batch_size)
                permuation2[indecies_not_unique] = temp_permutation[indecies_not_unique]

        # r_vec will contain [batch_size] random numbers between 0 to 1
        r_vec = np.random.uniform(size=batch_size)

        # one hot encoding
        indecies = np.arange(batch_size)
        one_hot1 = np.zeros((batch_size, self.number_of_classes))
        one_hot2 = np.zeros((batch_size, self.number_of_classes))

        one_hot1[indecies,labels[permuation1]] = 1
        one_hot2[indecies, labels[permuation2]] = 1

        labels_out = one_hot1 * r_vec[:, np.newaxis] + one_hot2 * (1.0 - r_vec[:, np.newaxis])

        # extract G1_vec, G2_vec

        fs = 44100
        a_weighting_chunk = int(0.1*fs)  # crop 0.1[sec] from the signal

        # padding data array, to get integer multiplication of a_weighting_chunk
        number_of_sections = int(np.ceil(data_array.shape[1]/a_weighting_chunk)+0.1)

        n_r = int(number_of_sections*a_weighting_chunk)

        zero_padding = n_r - data_array.shape[1]
        
        data_padded = np.pad(data_array, [(0, 0), (0, zero_padding)], mode='constant')

        data_padded_reshaped = data_padded.reshape((data_padded.shape[0]),a_weighting_chunk,number_of_sections)

        # filter coefficients
        b, a = A_weighting(fs)
        # A-weighting filtering
        y = lfilter(b, a, data_padded_reshaped, axis= 1)
        G_vec = np.max(np.max(y,1),1)

        G1_vec = 20*np.log10(1e-12+G_vec[permuation1])
        G2_vec = 20*np.log10(1e-12+G_vec[permuation2])

        p_denominator = 1 + ((1 - r_vec) / r_vec) * np.power(10, (G1_vec - G2_vec) / 20)
        p_vec = 1/p_denominator

        data1 = data_array[permuation1, :]
        data2 = data_array[permuation2, :]

        p_vec = p_vec[:, np.newaxis]
        data_out = (p_vec * data1 + (1 - p_vec) * data2) / np.sqrt(np.power(p_vec, 2) + np.power((1 - p_vec), 2))

        return data_out, labels_out
















