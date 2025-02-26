import tensorflow as tf

class TraderAgent():
    def __init__(self, window_size, num_sig, num_action=2):
        self.input_size = window_size*2
        self.num_sig = num_sig
        self.num_action = num_action
        self.model = self.model_builder()
        
    def model_builder(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(8,kernel_size=(3, 3), activation='relu', input_shape=(self.input_size,self.input_size,self.num_sig)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(16,kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=4, activation='relu'),
            tf.keras.layers.Dense(units=self.num_action, activation='linear')
        ])
        model.compile(loss='mse', optimizer='adam') # , metrics=['mse','mae']
        return model
           
    def train(self, samples,targets):
        self.model.fit(samples, targets, epochs=1, verbose=0) #, validation_data=(valSamplesX,valTargets)

    def test(self,test_samples):
        return self.model.predict(test_samples)
        