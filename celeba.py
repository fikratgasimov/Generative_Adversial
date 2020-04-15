import numpy as np
import keras

class celeba(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size = 4, dim = (32,32,32),
                 n_channels = 3, n_classes = 40, shuffle = True ):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    # length of dataset
    def __len__(self):
        # model sees training samples at most once per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    # then generate getitem in order to call batch sample
    # together with corresponding index
    def __getitem__(self, index):
        # Generate indexess of batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates index after each epoch"
        self.indexes = np.array(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):

        " Genearate data containing batch_size samples "
        #Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):

            # Store samples
            X[i, ] = np.load('data/', + ID + '.npy')
            # Store Classes
            y[i] = self.labels[ID]
        # Convert class vector to binary class vector
        return X, keras.utils.to_categorical(y, num_classes = self.n_classes)
