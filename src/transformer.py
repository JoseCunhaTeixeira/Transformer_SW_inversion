"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS, Sorbonne Université
Date : April 30, 2024
"""





from keras import Model
from keras import ops
from keras.layers import Input, Conv1D, Dense, MaxPooling1D
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
import json

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
    def __init__(self, params):
        self.params = params
        
        self.encoder_emb_dim = 64
        self.decoder_emb_dim = 64
        self.intermediate_dim = 256
        self.num_heads = 8
        self.encoder_N_layers = 4
        self.decoder_N_layers = 4

        self.model = self.build_model()
        self.id = self.generate_id()
        self.dump_params()


    def build_model(self):
        ### Feature Encoder -----------------------------------------------------------------------
        feature_encoder_inputs = Input(shape=(self.params['input_seq_format']['length'], 1))
        
        x1 = Conv1D(16, 3, activation='relu', padding='same')(feature_encoder_inputs)
        x1 = Conv1D(16, 3, activation='relu', padding='same')(x1)
        x1 = MaxPooling1D(pool_size=2)(x1)

        x1 = Conv1D(32, 3, activation='relu', padding='same')(x1)
        x1 = Conv1D(32, 3, activation='relu', padding='same')(x1)
        x1 = MaxPooling1D(pool_size=2)(x1)

        x1 = Conv1D(self.encoder_emb_dim, 3, activation='relu', padding='same')(x1)
        x1 = Conv1D(self.encoder_emb_dim, 3, activation='relu', padding='same')(x1)

        feature_encoder_outputs = x1
        ### ---------------------------------------------------------------------------------------


        ### Transformer Encoder -------------------------------------------------------------------
        # Positional Encoding
        position_encoding = SinePositionEncoding()(feature_encoder_outputs)
        x2 = feature_encoder_outputs + position_encoding

        # Encoder
        for _ in range(self.encoder_N_layers):
            x2 = TransformerEncoder(intermediate_dim=self.intermediate_dim, num_heads=self.num_heads)(inputs=x2)
        
        encoder_outputs = x2
        ### ---------------------------------------------------------------------------------------


        ### Transformer Decoder -------------------------------------------------------------------
        decoder_inputs = Input(shape=(None,))

        # Positional Encoding
        x3 = TokenAndPositionEmbedding(
            vocabulary_size=self.params['output_seq_format']['vocab_size'],
            sequence_length=self.params['output_seq_format']['length'],
            embedding_dim=self.decoder_emb_dim,
            # mask_zero=True,
        )(decoder_inputs)

        # Decoder
        for _ in range(self.decoder_N_layers):
            x3 = TransformerDecoder(intermediate_dim=self.intermediate_dim, num_heads=self.num_heads)(
                decoder_sequence=x3, encoder_sequence=encoder_outputs)

        decoder_outputs = Dense(self.params['output_seq_format']['vocab_size'], activation='softmax')(x3)
        ### ---------------------------------------------------------------------------------------

        
        ### Model ----------------------------------------------------------------------------------
        model = Model([feature_encoder_inputs, decoder_inputs], decoder_outputs, name='Transformer')
        model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        ### ---------------------------------------------------------------------------------------
    

        return model



    def generate_id(self):
        id = f'[{datetime.now().strftime("%Y%m%d%H%M")}]'

        folder_path = f'{PATH_MODELS}/{id}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return id
    


    def dump_params(self):
        folder_path = f'{PATH_MODELS}/{self.id}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if 'model_params' not in self.params:
            model_params = {
                'id': self.id,
                'encoder_emb_dim': self.encoder_emb_dim,
                'decoder_emb_dim': self.decoder_emb_dim,
                'intermediate_dim': self.intermediate_dim,
                'num_heads': self.num_heads,
                'encoder_N_layers': self.encoder_N_layers,
                'decoder_N_layers': self.decoder_N_layers,
                'trained': False,
                'training_number' : 0,
                'tested': False,
                'test_number' : 0
            }
            self.params['model_params'] = model_params 
        with open(f'{PATH_MODELS}/{self.id}/{self.id}_params.json', 'w') as f:
            json.dump(self.params, f, indent=2)
            
    
    
    def reload_params(self):
        with open(f'{PATH_MODELS}/{self.id}/{self.id}_params.json', 'r') as f:
            self.params = json.load(f)
        self.save_model()



    def train(self, X_train, y_train, epochs=50, batch_size=64, data_name=None, **kwargs):
        starting_date = datetime.now()

        if 'X_val' in kwargs and 'y_val' in kwargs:
            X_val = kwargs['X_val']
            y_val = kwargs['y_val']

            history = self.model.fit(
                [X_train, y_train[:, :-1, ...]], y_train[:, 1:, ...],
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=([X_val, y_val[:, :-1, ...]], y_val[:, 1:, ...]),
                callbacks=[EarlyStopping(monitor='val_loss', start_from_epoch=10, patience=10)],
                verbose=1
            )

        else:
            history = self.model.fit(
                [X_train, y_train[:, :-1, ...]], y_train[:, 1:, ...],
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=1
            )


        ending_date = datetime.now()
        training_time = ending_date - starting_date
        hours, remainder = divmod(training_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        training_params = {
            'data_name': data_name,
            'starting_date': starting_date.strftime('%Y-%m-%d %H:%M:%S'),
            'ending_date': ending_date.strftime('%Y-%m-%d %H:%M:%S'),
            'training_time': f'{hours} hours {minutes} minutes {seconds} seconds',
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0] if 'X_val' in kwargs and 'y_val' in kwargs else None,
            'batch_size': batch_size,
            'epochs': epochs,
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1] if 'X_val' in kwargs and 'y_val' in kwargs else None,
            'train_accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history['val_accuracy'][-1] if 'X_val' in kwargs and 'y_val' in kwargs else None
        }
        
        training_number = self.params['model_params']['training_number'] + 1
        self.params[f'training_params_{training_number}'] = training_params
        self.params['model_params']['trained'] = True
        self.params['model_params']['training_number'] = training_number
        self.dump_params()


        self.plot_training_history(history, **kwargs)



    def plot_training_history(self, history, **kwargs):
        fig, ax = plt.subplots(figsize=(16, 10), dpi=300)

        # Plot Loss
        ax.plot(history.history['loss'])
        if 'X_val' in kwargs and 'y_val' in kwargs:
            ax.plot(history.history['val_loss'])
            ax.legend(['Training dataset', 'Validation dataset'])
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Categorical Crossentropy Loss')

        # Plot Accuracy
        ax_twin = ax.twinx()
        ax_twin.plot(history.history['accuracy'], linestyle='--')
        if 'X_val' in kwargs and 'y_val' in kwargs:
            ax_twin.plot(history.history['val_accuracy'], linestyle='--')
        ax_twin.set_xlabel('Epochs')
        ax_twin.set_ylabel('Accuracy')

        training_number = self.params['model_params']['training_number']
        fig.savefig(f'{PATH_MODELS}/{self.id}/{self.id}_training_{training_number}_history.png', format='png', dpi='figure', bbox_inches='tight')



    def save_model(self):
        with open(f'{PATH_MODELS}/{self.id}/{self.id}_model.pkl', 'wb') as f:
            dump(self, f)



    def decode_seq(self, input_seq):
        input_seq = ops.convert_to_tensor(input_seq)

        def next(prompt, cache, index):
            logits = self.model([input_seq, prompt])[:, index-1, :]
            hidden_states = None
            return logits, hidden_states, cache

        prompt = [self.params['output_seq_format']['word_to_index']['[START]']]
        while len(prompt) < self.params['output_seq_format']['length']:
            prompt.append(self.params['output_seq_format']['word_to_index']['[PAD]'])
        prompt = np.array(prompt).reshape(1, len(prompt))
        prompt = ops.convert_to_tensor(prompt)

        decoded_seq = GreedySampler()(
            next,
            prompt,
            stop_token_ids=[self.params['output_seq_format']['word_to_index']['[END]']],
            index=1
        )

        return ops.convert_to_numpy(decoded_seq)[0][1:]
    


    def decode_seq_restrictive(self, input_seq):
        input_seq = ops.convert_to_tensor(input_seq)

        def next(prompt, cache, index):
            logits = self.model([input_seq, prompt])[:, index-1, :]
            hidden_states = None
            return logits, hidden_states, cache

        prompt = [self.params['output_seq_format']['word_to_index']['[START]']]
        while len(prompt) < self.params['output_seq_format']['length']:
            prompt.append(self.params['output_seq_format']['word_to_index']['[PAD]'])
        prompt = np.array(prompt).reshape(1, len(prompt))
        prompt = ops.convert_to_tensor(prompt)

        decoded_seq  = RestrictiveSampler(self.params['output_seq_format']['forbidden_tokens'])(
            next,
            prompt,
            stop_token_ids=[self.params['output_seq_format']['word_to_index']['[END]']],
            index=1
            )

        return ops.convert_to_numpy(decoded_seq)[0][1:]
    


    def evaluate(self, X_test, y_test, data_name=None):
        starting_date = datetime.now()

        words = self.params['output_seq_format']['vocab']
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

        ending_date = datetime.now()

        print(f'\nAccuracy: {int(accuracy*100)} %')
        print(f'Precision: {int(precision*100)} %')
        print(f'Recall: {int(recall*100)} %')
        print(f'F1: {int(F1*100)} %')


        folder_path = f'{PATH_MODELS}/{self.id}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        test_params = {
            'data_name': data_name,
            'starting_date': starting_date.strftime('%Y-%m-%d %H:%M:%S'),
            'ending_date': ending_date.strftime('%Y-%m-%d %H:%M:%S'),
            'test_samples': X_test.shape[0],
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'F1': float(1)            
        }
        for i, word in enumerate(words):
            test_params[word] = {
                'precision': None if np.isnan(precisions[i]) else precisions[i],
                'recall': None if np.isnan(recalls[i]) else recalls[i],
                'F1': None if np.isnan(F1s[i]) else F1s[i]
            }
        
        test_number = self.params['model_params']['test_number'] + 1
        self.params[f'test_params_{test_number}'] = test_params
        self.params['model_params']['tested'] = True
        self.params['model_params']['test_number'] = test_number
        self.dump_params()


        fig, ax = plt.subplots(dpi=300, figsize=(16, 9))
        ax.scatter(range(len(accuracies)), accuracies)
        ax.set_xlabel('Sample (#)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim([0,1])
        fig.savefig(f'{PATH_MODELS}/{self.id}/{self.id}_test{test_number}_accuracies.png', format='png', dpi='figure', bbox_inches='tight')

        fig, ax = plt.subplots(dpi=300, figsize=(16, 9))
        heatmap(conf_matrix, annot=True, cmap='Reds', ax=ax, xticklabels=words, yticklabels=words)
        ax.set_xlabel('Decoded Words')
        ax.set_ylabel('Expected Words')
        ax.set_title('Confusion Matrix')
        fig.savefig(f'{PATH_MODELS}/{self.id}/{self.id}_test{test_number}_confusion_matrix.png', format='png', dpi='figure', bbox_inches='tight')
        ### -----------------------------------------------------------------------------------------------