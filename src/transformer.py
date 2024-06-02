"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""





from keras import Model
from keras import ops
from keras.layers import Input, Conv1D, MaxPooling1D, Dense
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.metrics import Accuracy
from keras_nlp.layers import SinePositionEncoding, TokenAndPositionEmbedding, TransformerEncoder, TransformerDecoder
from keras_nlp.samplers import GreedySampler, Sampler

import os
import numpy as np
from tqdm import tqdm
from seaborn import heatmap
from pickle import dump
from datetime import datetime
import matplotlib.pyplot as plt

from folders import PATH_MODELS





class RestrictiveSampler(Sampler):
  def __init__(self, forbidden_tokens, **kwargs):
      super().__init__(**kwargs)
      self.forbidden_tokens = forbidden_tokens
      self.current_seq_index = 0

  def get_next_token(self, probs):
      batch_size, vocab_size = ops.shape(probs)
      for id in self.forbidden_tokens[self.current_seq_index]:
          update = ops.zeros((batch_size, 1))
          probs = ops.slice_update(probs, (0, id), update)
      self.current_seq_index += 1
      return ops.argmax(probs, axis=-1)





class Transformer:
    def __init__(self, input_sequence_format, output_sequence_format):

        self.input_sequence_format = input_sequence_format
        self.output_sequence_format = output_sequence_format

        self.encoder_emb_dim = 64
        self.decoder_emb_dim = 64
        

        self.intermediate_dim = 256
        self.num_heads = 8
        
        self.encoder_N_layers = 4
        self.decoder_N_layers = 4
    

        self.model = self.build_model()

        self.id = self.generate_id()

        self.save_specs()



    def build_encoder(self):
        encoder_inputs = Input(shape=(self.input_sequence_format.length, 1))

        c1 = Conv1D(16, 3, activation='relu', padding='same')(encoder_inputs)
        c1 = Conv1D(16, 3, activation='relu', padding='same')(c1)
        p1 = MaxPooling1D(pool_size=2)(c1)

        c2 = Conv1D(32, 3, activation='relu', padding='same')(p1)
        c2 = Conv1D(32, 3, activation='relu', padding='same')(c2)
        p2 = MaxPooling1D(pool_size=2)(c2)

        c3 = Conv1D(64, 3, activation='relu', padding='same')(p2)
        c3 = Conv1D(64, 3, activation='relu', padding='same')(c3)

        position_encoding = SinePositionEncoding()(c3)
        x1 = c3 + position_encoding

        for _ in range(self.encoder_N_layers):
            x1 = TransformerEncoder(intermediate_dim=self.intermediate_dim, num_heads=self.num_heads)(inputs=x1)

        encoder_outputs = x1

        encoder = Model(encoder_inputs, encoder_outputs)

        return encoder, encoder_inputs, encoder_outputs



    def build_decoder(self):
        decoder_inputs = Input(shape=(None,))
        encoded_seq_inputs = Input(shape=(None, self.encoder_emb_dim))

        x2 = TokenAndPositionEmbedding(
            vocabulary_size=self.output_sequence_format.vocab.size,
            sequence_length=self.output_sequence_format.length,
            embedding_dim=self.decoder_emb_dim,
            # mask_zero=True,
        )(decoder_inputs)

        for _ in range(self.decoder_N_layers):
            x2 = TransformerDecoder(intermediate_dim=self.intermediate_dim, num_heads=self.num_heads)(
                decoder_sequence=x2, encoder_sequence=encoded_seq_inputs)

        decoder_outputs = Dense(self.output_sequence_format.vocab.size, activation='softmax')(x2)
        decoder = Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

        return decoder, decoder_inputs, decoder_outputs



    def build_model(self):
        _, encoder_inputs, encoder_outputs = self.build_encoder()
        decoder, decoder_inputs, decoder_outputs = self.build_decoder()

        decoder_outputs = decoder([decoder_inputs, encoder_outputs])

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='Transformer')

        model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        
        return model
    


    def generate_id(self):
        id = f'[{datetime.now().strftime("%Y%m%d%H%M")}]'

        folder_path = f'{PATH_MODELS}/{id}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return id
    

    def save_specs(self):
        folder_path = f'{PATH_MODELS}/{self.id}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(f'{PATH_MODELS}/{self.id}/{self.id}_model_specs.txt', 'w') as f:
            f.write(f'Model ID: {self.id}\n\n')
            f.write(f'Encoder Embedding Dimension: {self.encoder_emb_dim}\n')
            f.write(f'Decoder Embedding Dimension: {self.decoder_emb_dim}\n')
            f.write(f'Intermediate Dimension: {self.intermediate_dim}\n')
            f.write(f'Number of Heads: {self.num_heads}\n')
            f.write(f'Encoder Number of Layers: {self.encoder_N_layers}\n')
            f.write(f'Decoder Number of Layers: {self.decoder_N_layers}\n')
            f.write(f'Input Sequence Length: {self.input_sequence_format.length}\n')
            f.write(f'Output Sequence Length: {self.output_sequence_format.length}\n')



    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        starting_date = datetime.now()

        history = self.model.fit(
            [X_train, y_train[:, :-1]], y_train[:, 1:],
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=([X_val, y_val[:, :-1]], y_val[:, 1:]),
            callbacks=[EarlyStopping(monitor='val_loss', start_from_epoch=10, patience=10)],
            verbose=1
        )

        ending_date = datetime.now()
        training_time = ending_date - starting_date
        hours, remainder = divmod(training_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)


        folder_path = f'{PATH_MODELS}/{self.id}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(f'{PATH_MODELS}/{self.id}/{self.id}_training_info.txt', 'w') as f:
            f.write(f'Model ID: {self.id}\n\n')
            f.write(f'Training starting date: {starting_date}\n')
            f.write(f'Training ending date: {ending_date}\n')
            f.write(f'Training Time: {hours} hours {minutes} minutes {seconds} seconds\n')
            f.write(f'Train samples: {X_train.shape[0]}\n')
            f.write(f'Validation samples: {X_val.shape[0]}\n')
            f.write(f'Batch size: {batch_size}\n')
            f.write(f'Epochs: {len(history.history["loss"])}\n')
            f.write(f'Training Loss: {history.history["loss"][-1]}\n')
            f.write(f'Validation Loss: {history.history["val_loss"][-1]}\n')
            f.write(f'Training Accuracy: {history.history["accuracy"][-1]}\n')
            f.write(f'Validation Accuracy: {history.history["val_accuracy"][-1]}\n')

        self.plot_training_history(history)



    def plot_training_history(self, history):
        fig, ax = plt.subplots(figsize=(16, 10), dpi=300)

        # Plot Loss
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Categorical Crossentropy Loss')
        ax.legend(['Training dataset', 'Validation dataset'])

        # Plot Accuracy
        ax_twin = ax.twinx()
        ax_twin.plot(history.history['accuracy'], linestyle='--')
        ax_twin.plot(history.history['val_accuracy'], linestyle='--')
        ax_twin.set_xlabel('Epochs')
        ax_twin.set_ylabel('Accuracy')

        fig.savefig(f'{PATH_MODELS}/{self.id}/{self.id}_training_history.png', format='png', dpi='figure', bbox_inches='tight')



    def save_model(self):
        with open(f'{PATH_MODELS}/{self.id}/{self.id}_model.pkl', 'wb') as f:
            dump(self, f)



    def decode_seq(self, input_seq):
        input_seq = ops.convert_to_tensor(input_seq)

        def next(prompt, cache, index):
            logits = self.model([input_seq, prompt])[:, index-1, :]
            hidden_states = None
            return logits, hidden_states, cache

        prompt = [self.output_sequence_format.vocab.word_to_index['[START]']]
        while len(prompt) < self.output_sequence_format.length:
            prompt.append(self.output_sequence_format.vocab.word_to_index['[PAD]'])
        prompt = np.array(prompt).reshape(1, len(prompt))
        prompt = ops.convert_to_tensor(prompt)

        decoded_seq = GreedySampler()(
            next,
            prompt,
            stop_token_ids=[self.output_sequence_format.vocab.word_to_index['[END]']],
            index=1
        )

        return ops.convert_to_numpy(decoded_seq)[0][1:]
    


    def decode_seq_restrictive(self, input_seq):
        input_seq = ops.convert_to_tensor(input_seq)

        def next(prompt, cache, index):
            logits = self.model([input_seq, prompt])[:, index-1, :]
            hidden_states = None
            return logits, hidden_states, cache

        prompt = [self.output_sequence_format.vocab.word_to_index['[START]']]
        while len(prompt) < self.output_sequence_format.length:
            prompt.append(self.output_sequence_format.vocab.word_to_index['[PAD]'])
        prompt = np.array(prompt).reshape(1, len(prompt))
        prompt = ops.convert_to_tensor(prompt)

        decoded_seq  = RestrictiveSampler(self.output_sequence_format.forbidden_tokens)(
            next,
            prompt,
            stop_token_ids=[self.output_sequence_format.vocab.word_to_index['[END]']],
            index=1
            )

        return ops.convert_to_numpy(decoded_seq)[0][1:]
    


    def evaluate(self, X_test, y_test):
        words = self.output_sequence_format.vocab.words.copy()
        conf_matrix = np.zeros((len(words), len(words)))

        accuracies = []

        for i_sample in tqdm(range(X_test.shape[0])):
            input_seq = X_test[i_sample, ...].reshape(1, X_test.shape[1], 1)
            target_seq = y_test[i_sample, 1:-1]

            # decoded_seq = self.decode_seq(input_seq)
            decoded_seq = self.decode_seq_restrictive(input_seq)

            m = Accuracy()
            accuracy = m(target_seq, decoded_seq).numpy()
            accuracies.append(accuracy)

            for decoded_word_idx, target_word_idx in zip(decoded_seq, target_seq):
                conf_matrix[target_word_idx, decoded_word_idx] += 1


        accuracies = np.array(accuracies)
        accuracy = np.nanmean(accuracies)


        precisions = []
        recalls = []
        F1s = []
        for i in range(len(words)):
            true_positives = conf_matrix[i, i]
            false_positives = np.sum(conf_matrix[:, i]) - true_positives
            false_negatives = np.sum(conf_matrix[i, :]) - true_positives

            if true_positives == 0 and false_positives == 0:
                precisions.append(np.nan)
            else:
                precisions.append(true_positives / (true_positives + false_positives))
            
            if true_positives == 0 and false_negatives == 0:
                recalls.append(np.nan)
            else:
                recalls.append(true_positives / (true_positives + false_negatives))
            
            if true_positives == 0 and false_positives == 0 and false_negatives == 0:
                F1s.append(np.nan)
            else:
                F1s.append(2 * (precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1]))

        precision = np.nanmean(precisions)
        recall = np.nanmean(recalls)
        F1 = np.nanmean(F1s)

        
        print(f'\nAccuracy: {int(accuracy*100)} %')
        print(f'Precision: {int(precision*100)} %')
        print(f'Recall: {int(recall*100)} %')
        print(f'F1: {int(F1*100)} %')


        folder_path = f'{PATH_MODELS}/{self.id}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(f'{PATH_MODELS}/{self.id}/{self.id}_test_info.txt', 'w') as f:
            f.write(f'Model ID: {self.id}\n\n')
            f.write(f'Test ending date: {datetime.now()}\n')
            f.write(f'Test samples: {X_test.shape[0]}\n')
            f.write(f'Accuracy: {accuracy:.2f}\n')
            f.write(f'Precision: {precision:.2f}\n')
            f.write(f'Recall: {recall:.2f}\n')
            f.write(f'F1: {F1:.2f}\n\n')

            for i, word in enumerate(words):
                f.write(f'\n{word}:\n')
                f.write(f'Precision: {precisions[i]:.2f}\n')
                f.write(f'Recall: {recalls[i]:.2f}\n')
                f.write(f'F1: {F1s[i]:.2f}\n')


        fig, ax = plt.subplots(dpi=300, figsize=(16, 9))
        ax.scatter(range(len(accuracies)), accuracies)
        ax.set_xlabel('Sample (#)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim([0,1])
        fig.savefig(f'{PATH_MODELS}/{self.id}/{self.id}_test_accuracies.png', format='png', dpi='figure', bbox_inches='tight')

        fig, ax = plt.subplots(dpi=300, figsize=(16, 9))
        heatmap(conf_matrix, annot=True, cmap='Reds', ax=ax, xticklabels=words, yticklabels=words)
        ax.set_xlabel('Decoded Words')
        ax.set_ylabel('Expected Words')
        ax.set_title('Confusion Matrix')
        fig.savefig(f'{PATH_MODELS}/{self.id}/{self.id}_test_confusion_matrix.png', format='png', dpi='figure', bbox_inches='tight')
        ### -----------------------------------------------------------------------------------------------