from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, Dropout

embed1 = Sequential()
embed1.add(Embedding())
embed2 = Sequential()
embed2.add(Embedding())
