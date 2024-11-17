# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

The goal of this project is to identify entities such as people, places, and organizations in text by implementing an LSTM-based model for named entity recognition (NER). Using deep learning techniques, our goal is to create a reliable system that can correctly identify named things in unstructured text data.

## DESIGN STEPS

# STEP 1:
Import the necessary packages.

# STEP 2:
Read the dataset, and fill the null values using forward fill.

# STEP 3:
Create a list of words and tags. Then, find the number of unique words and tags in the dataset.

# STEP 4:
Create a dictionary for the words and their Index values. Do the same for the tags as well, Now we move to moulding the data for training and testing.

# STEP 5:
We do this by padding the sequences, This is done to achieve the same length of input data

# STEP 6:
We build a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, and time-distributed dense Layers.

# STEP 7:
We compile the model and fit the train and validation sets. We plot the necessary graphs for analysis. A custom prediction is made to test the model manually.

## PROGRAM
### Name: Gayathri A
### Register Number: 212221230028
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model

data = pd.read_csv("ner_dataset.csv", encoding="latin1")

data.head(50)

data = data.fillna(method="ffill")

data.head(50)

print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())

words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())

print("Unique tags are:", tags)

num_words = len(words)
num_tags = len(tags)

num_words

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences

len(sentences)

sentences[0]

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

word2idx

plt.hist([len(s) for s in sentences], bins=50)
plt.show()

X1 = [[word2idx[w[0]] for w in s] for s in sentences]

type(X1[0])

X1[0]

max_len = 50

X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)

X[0]

y1 = [[tag2idx[w[2]] for w in s] for s in sentences]

y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)

X_train[0]

y_train[0]

input_word = layers.Input(shape=(max_len,))
embedding_layer = layers.Embedding(input_dim=num_words,output_dim=50,
                                   input_length=max_len)(input_word)

dropout_layer = layers.SpatialDropout1D(0.1)(embedding_layer)

bidirection_lstm = layers.Bidirectional(
    layers.LSTM(units=100,return_sequences=True,
                recurrent_dropout=0.1))(dropout_layer)

output = layers.TimeDistributed(
    layers.Dense(num_tags,activation="softmax"))(bidirection_lstm)

model = Model(input_word, output)

print('Name:Gayathri \t\tRegister Number:212221230028\n')
model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test,y_test),
    batch_size=32,
    epochs=3,
)

metrics = pd.DataFrame(model.history.history)
metrics.head()

print('Name:Gayathri \t\tRegister Number:212221230028\n')
metrics[['accuracy','val_accuracy']].plot()

print('Name:Gayathri \t\tRegister Number:212221230028\n')
metrics[['loss','val_loss']].plot()

i = 20
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print('Name:Gayathri \t\tRegister Number:212221230028\n')
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/ad3d2176-defc-4add-aad3-2c3c90d615a4)

### Sample Text Prediction

![Screenshot 2024-11-17 142044](https://github.com/user-attachments/assets/1e5fc170-af49-4aec-8e54-513b090629a0)



![image](https://github.com/user-attachments/assets/75c52902-9ad2-47f9-838a-b49aec54a934)

![image](https://github.com/user-attachments/assets/fcef0f1e-6da8-4e55-88e1-a1f104e986a4)

![image](https://github.com/user-attachments/assets/2f649829-2aac-45f9-a351-8c6973627fb8)


## RESULT

Thus, an LSTM-based model (bi-directional) for recognizing the named entities in the text is developed Successfully.
