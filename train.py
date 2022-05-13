import os
from argparse import ArgumentParser
import time
import tensorflow as tf
from data import Dataset
import numpy as np
from model.layers.tranformers_xl import *

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--d-ff", default=512, type=int)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--embedding-size", default=128, type=int)
    parser.add_argument("--n-layer", default=6, type=int)
    parser.add_argument("--max-len", default=128, type=int)
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--buffer-size", default=128, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--data-path", default="/content/transformer-xl1/data/clean/clean_data.csv", type=str)

    args = parser.parse_args()


    home_dir = os.getcwd()
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${accout}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ${name} model with hyper-params:') # FIXME
    print('===========================')
    
    # FIXME
    # Do Prediction

dataset = Dataset(args.data_path)

train_dataset, val_dataset = dataset.build_dataset(max_length=args.max_len, vocab_size=args.vocab_size, 
                                                    test_size = args.test_size, buffer_size=args.buffer_size,
                                                    batch_size= args.batch_size, cleaned_data=True)

sentences_tokenizer = dataset.tokenizer_save
sentences_tokenizer_size = len(sentences_tokenizer.word_counts) + 1


def cal_acc(real, pred):
        pred_labels = tf.math.argmax(pred, 1)
        acc = np.sum(np.equal(pred_labels, real))/len(real)
        return acc

class CustomLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 500):
        # TODO: Update document
        super(CustomLearningRate, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        
    def __call__(self, step_num):
        # TODO: Update document
        lrate = tf.cast(self.d_model, tf.float32) ** (-0.5) * tf.math.minimum(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5) )
        return lrate

model = TransformerXL(n_vocab=sentences_tokenizer_size,
                        d_embed=args.embedding_size,
                        d_model=args.d_model,
                        d_ff=args.d_ff,
                        q_len=args.max_len,
                        m_len=args.max_len,
                        num_heads=args.num_heads,
                        n_layer=args.n_layer,
                        dropout_rate=args.dropout)

learning_rate = CustomLearningRate(args.d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# train_loss = tf.keras.metrics.Mean(name='train_loss')
train_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
train_acc = tf.keras.metrics.CategoricalAccuracy()
# train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
checkpoint = tf.train.Checkpoint(model=model, optimizer = optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, '/content', max_to_keep=3)

# def loss_function(labels, logits):
#     """
#     Compute the loss function
#     """
#     loss = train_loss(
#         labels, logits)
#     # loss = tf.reduce_mean(loss)
#     return loss

def train_step(inputs, labels, optimizer, inputs_mem):
    with tf.GradientTape() as tape:
        # print(labels.shape)
        labels1 = tf.keras.utils.to_categorical(labels, 2)
        # print(labels.shape)
        logits, new_mems = model(inputs, inputs_mem, training=True)
        # print(logits.shape)
        x = tf.keras.layers.GlobalAveragePooling1D()(logits)
        x = tf.keras.layers.Dropout(0.1)(x, training=True)
        x = tf.keras.layers.Dense(20, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x, training=True)
        logits = tf.keras.layers.Dense(2, activation="softmax")(x)
        # loss = loss_function(labels1, logits)
        loss = train_loss(labels1, logits)
        
    #compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    #update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # train_loss.update_state(loss)
    train_acc.update_state(labels, logits)

    return new_mems, loss
    
def fit():

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    mems=None

    for epoch in range(args.epochs):
        
        for (epoch, (inputs, labels)) in enumerate(train_dataset):

            mems, loss = train_step(inputs, labels, optimizer, mems)

        print('{} epoch: {} | loss: {:.4f} | acc: {:.4f}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    epoch,
                    loss,
                    train_acc.result()))

        if (epoch + 1) % 5 == 0:
            saved_path = checkpoint_manager.save()
            print('saving checkpoint for epoch {} at {}'.format(
                    epoch+1, saved_path))

        # train_loss.reset_states()
        train_acc.reset_states()
        
    print('----------------Done--------------------')

# def predict(sentence):
#     sentence = sentence.lower()
#     sentence = sentence.split()
#     sentence = dataset.tokenizer.texts_to_sequences([sentence])
#     sentence = tf.keras.preprocessing.sequence.pad_sequences(sentence, maxlen = args.max_len, padding='post')
#     sentence = tf.convert_to_tensor(sentence)
#     sentence = tf.expand_dims(sentence, 0)
#     sentence = tf.expand_dims(sentence, -1)
#     sentence = tf.cast(sentence, tf.float32)
#     mems = None
#     logits, new_mems = model(sentence, mems, training=False)
#     pred = tf.math.argmax(logits, 1)
#     return pred


fit()   



