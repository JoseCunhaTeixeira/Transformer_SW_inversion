"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""





from numpy import arange





class SoilVocab:
  def __init__(self):
    self.words = self.make_vocab()
    self.size = len(self.words)
    self.max_depth = 20

    self.word_to_index, self.index_to_word = self.make_index_representation()

  def make_vocab(self):
    vocab = [
        '[PAD]',
        '[START]',
        '[END]',

        '[WT]',

        '[SOIL1]',
        '[SOIL2]',
        '[SOIL3]',
        '[SOIL4]',

        '[THICKNESS1]',
        '[THICKNESS2]',
        '[THICKNESS3]',
        '[THICKNESS4]',

        '[N1]',
        '[N2]',
        '[N3]',
        '[N4]',

        '0.0',
        '0.5',
        '1.0',
        '1.5',
        '2.0',
        '2.5',
        '3.0',
        '3.5',
        '4.0',
        '4.5',
        '5.0',
        '5.5',
        '6.0',
        '6.5',
        '7.0',
        '7.5',
        '8.0',
        '8.5',
        '9.0',
        '9.5',
        '10.0',
        '10.5',
        '11.0',
        '11.5',
        '12.0',
        '12.5',
        '13.0',
        '13.5',
        '14.0',
        '15.0',
        '16.0',
        '17.0',
        '18.0',
        '19.0',
        '20.0',

        'clay',
        'silt',
        'loam',
        'sand',
        ]
    return vocab
   
  def make_index_representation(self):
    word_to_index = {}
    for i, word in enumerate(self.words):
        word_to_index[word] = i
    index_to_word = {index : word for word, index in word_to_index.items()}
    return word_to_index, index_to_word



class OutputSequenceFormat:
  def __init__(self):
    self.vocab = SoilVocab()
    self.length = 28 #20

    self.allowed_tokens = self.make_allowed_tokens()
    self.forbidden_tokens = self.make_forbidden_tokens()

  def make_allowed_tokens(self):
    allowed_tokens = [
                  [self.vocab.word_to_index['[WT]']],

                  [self.vocab.word_to_index['0.5'],
                   self.vocab.word_to_index['1.0'],
                   self.vocab.word_to_index['1.5'],
                   self.vocab.word_to_index['2.0'],
                   self.vocab.word_to_index['2.5'],
                   self.vocab.word_to_index['3.0'],
                   self.vocab.word_to_index['3.5'],
                   self.vocab.word_to_index['4.0'],
                   self.vocab.word_to_index['4.5'],
                   self.vocab.word_to_index['5.0'],
                   self.vocab.word_to_index['5.5'],
                   self.vocab.word_to_index['6.0'],
                   self.vocab.word_to_index['6.5'],
                   self.vocab.word_to_index['7.0'],
                   self.vocab.word_to_index['7.5'],
                   self.vocab.word_to_index['8.0'],
                   self.vocab.word_to_index['8.5'],
                   self.vocab.word_to_index['9.0'],
                   self.vocab.word_to_index['9.5']],

                  [self.vocab.word_to_index['[SOIL1]']],

                  [self.vocab.word_to_index['clay'],
                   self.vocab.word_to_index['silt'],
                   self.vocab.word_to_index['loam'],
                   self.vocab.word_to_index['sand']],

                  [self.vocab.word_to_index['[THICKNESS1]']],

                  [self.vocab.word_to_index['1.0'],
                   self.vocab.word_to_index['2.0'],
                   self.vocab.word_to_index['3.0'],
                   self.vocab.word_to_index['4.0'], 
                   self.vocab.word_to_index['5.0'],
                   self.vocab.word_to_index['6.0'],
                   self.vocab.word_to_index['7.0'],
                   self.vocab.word_to_index['8.0'],
                   self.vocab.word_to_index['9.0'],
                   self.vocab.word_to_index['10.0'],
                   self.vocab.word_to_index['11.0'],
                   self.vocab.word_to_index['12.0'],
                   self.vocab.word_to_index['13.0'],
                   self.vocab.word_to_index['14.0'],
                   self.vocab.word_to_index['15.0'],
                   self.vocab.word_to_index['16.0'],
                   self.vocab.word_to_index['17.0'],
                   self.vocab.word_to_index['18.0'],
                   self.vocab.word_to_index['19.0'], 
                   self.vocab.word_to_index['20.0']],

                  [self.vocab.word_to_index['[N1]']],
                   
                  [self.vocab.word_to_index['6.0'],
                   self.vocab.word_to_index['7.0'],
                   self.vocab.word_to_index['8.0'],
                   self.vocab.word_to_index['9.0'],
                   self.vocab.word_to_index['10.0']],

                  [self.vocab.word_to_index['[SOIL2]'],
                   self.vocab.word_to_index['[END]']],

                  [self.vocab.word_to_index['clay'],
                   self.vocab.word_to_index['silt'],
                   self.vocab.word_to_index['loam'],
                   self.vocab.word_to_index['sand']],

                  [self.vocab.word_to_index['[THICKNESS2]']],

                  [self.vocab.word_to_index['1.0'],
                   self.vocab.word_to_index['2.0'],
                   self.vocab.word_to_index['3.0'],
                   self.vocab.word_to_index['4.0'],
                   self.vocab.word_to_index['5.0'],
                   self.vocab.word_to_index['6.0'],
                   self.vocab.word_to_index['7.0'],
                   self.vocab.word_to_index['8.0'],
                   self.vocab.word_to_index['9.0'],
                   self.vocab.word_to_index['10.0'],
                   self.vocab.word_to_index['11.0'],
                   self.vocab.word_to_index['12.0'], 
                   self.vocab.word_to_index['13.0'],
                   self.vocab.word_to_index['14.0'],
                   self.vocab.word_to_index['15.0'],
                   self.vocab.word_to_index['16.0'],
                   self.vocab.word_to_index['17.0'], 
                   self.vocab.word_to_index['18.0'],
                   self.vocab.word_to_index['19.0'],
                   self.vocab.word_to_index['20.0']],

                  [self.vocab.word_to_index['[N2]']],
                   
                  [self.vocab.word_to_index['6.0'],
                   self.vocab.word_to_index['7.0'],
                   self.vocab.word_to_index['8.0'],
                   self.vocab.word_to_index['9.0'],
                   self.vocab.word_to_index['10.0']],

                  [self.vocab.word_to_index['[SOIL3]'],
                   self.vocab.word_to_index['[END]']],

                  [self.vocab.word_to_index['clay'],
                   self.vocab.word_to_index['silt'],
                   self.vocab.word_to_index['loam'],
                   self.vocab.word_to_index['sand']],

                  [self.vocab.word_to_index['[THICKNESS3]']],

                  [self.vocab.word_to_index['1.0'],
                   self.vocab.word_to_index['2.0'],
                   self.vocab.word_to_index['3.0'],
                   self.vocab.word_to_index['4.0'],
                   self.vocab.word_to_index['5.0'],
                   self.vocab.word_to_index['6.0'], 
                   self.vocab.word_to_index['7.0'],
                   self.vocab.word_to_index['8.0'],
                   self.vocab.word_to_index['9.0'],
                   self.vocab.word_to_index['10.0'],
                   self.vocab.word_to_index['11.0'],
                   self.vocab.word_to_index['12.0'], 
                   self.vocab.word_to_index['13.0'],
                   self.vocab.word_to_index['14.0'],
                   self.vocab.word_to_index['15.0'],
                   self.vocab.word_to_index['16.0'],
                   self.vocab.word_to_index['17.0'],
                   self.vocab.word_to_index['18.0'],
                   self.vocab.word_to_index['19.0'],
                   self.vocab.word_to_index['20.0']],

                  [self.vocab.word_to_index['[N3]']],
                   
                  [self.vocab.word_to_index['6.0'],
                   self.vocab.word_to_index['7.0'],
                   self.vocab.word_to_index['8.0'],
                   self.vocab.word_to_index['9.0'],
                   self.vocab.word_to_index['10.0']],

                  [self.vocab.word_to_index['[SOIL4]'],
                   self.vocab.word_to_index['[END]']],

                  [self.vocab.word_to_index['clay'],
                   self.vocab.word_to_index['silt'],
                   self.vocab.word_to_index['loam'],
                   self.vocab.word_to_index['sand']],

                  [self.vocab.word_to_index['[THICKNESS4]']],

                  [self.vocab.word_to_index['1.0'],
                   self.vocab.word_to_index['2.0'],
                   self.vocab.word_to_index['3.0'],
                   self.vocab.word_to_index['4.0'],
                   self.vocab.word_to_index['5.0'], 
                   self.vocab.word_to_index['6.0'],
                   self.vocab.word_to_index['7.0'],
                   self.vocab.word_to_index['8.0'],
                   self.vocab.word_to_index['9.0'],
                   self.vocab.word_to_index['10.0'],
                   self.vocab.word_to_index['11.0'],
                   self.vocab.word_to_index['12.0'],
                   self.vocab.word_to_index['13.0'],
                   self.vocab.word_to_index['14.0'],
                   self.vocab.word_to_index['15.0'],
                   self.vocab.word_to_index['16.0'],
                   self.vocab.word_to_index['17.0'],
                   self.vocab.word_to_index['18.0'],
                   self.vocab.word_to_index['19.0'],
                   self.vocab.word_to_index['20.0']],

                  [self.vocab.word_to_index['[N4]']],
                   
                  [self.vocab.word_to_index['6.0'],
                   self.vocab.word_to_index['7.0'],
                   self.vocab.word_to_index['8.0'],
                   self.vocab.word_to_index['9.0'],
                   self.vocab.word_to_index['10.0']],

                  [self.vocab.word_to_index['[END]']],
                ]
    return allowed_tokens
  
  def make_forbidden_tokens(self):
    tokens = list(self.vocab.word_to_index.values())
    forbidden_tokens = []
    for i in range(0, self.length-1):
      forbidden_tokens.append(tokens.copy())
    for i in range(0, self.length-1):
      for j in range(len(self.allowed_tokens[i])):
        forbidden_tokens[i].remove(self.allowed_tokens[i][j])
    return forbidden_tokens





class Frequencies:
  def __init__(self):
    self.min = 20
    self.max = 50
    self.step = 1
    self.values = arange(self.min, self.max+self.step, self.step)
    self.size = len(self.values)



class InputSequenceFormat():
  def __init__(self):
    self.vocab = Frequencies()
    self.length = self.vocab.size