# from msilib.schema import CustomAction
# from pickletools import optimize
# from pyexpat import model
# from sklearn.model_selection import learning_curve
import tensorflow as tf
from data import Dataset
# from model import TransformerXL
# from data import *
from tranformers_xl import *
vocab_size = 1000  # Only consider the top 20k words
maxlen = 16  # Only consider the first 200 words of each movie review

# def load_dataset():
#     (x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
#     return (x_train, y_train), (x_val, y_val)

# train_set, val_set = load_dataset()

# x_train = tf.keras.preprocessing.sequence.pad_sequences(train_set[0], maxlen=maxlen)
# x_val = tf.keras.preprocessing.sequence.pad_sequences(val_set[0], maxlen=maxlen)
# word embeeding size
EMBEDDING_SIZE = 128
# multihead attetion hidden size
HIDDEN_SIZE = 128
# feed forward network hidden size
FFN_SIZE = 512
# number of heads of multiheads
NUM_HEADS = 8
# target length, or sequence length
SEQ_LEN = 16
# memory length
MEM_LEN = 32
# number of layers of multihead attention
N_LAYER = 6
DROPOUT_RATE = 0.1
# wheather the bias of each layer of relative multihead attention is different or not
UNTIE_REL_BIAS = True

data_path = '/content/Transformer-XL/data/clean/clean_data.csv'

dataset = Dataset(data_path)
train_dataset, val_dataset = dataset.build_dataset(cleaned_data=True)
sentences_tokenizer = dataset.tokenizer_save
sentences_tokenizer_size = len(sentences_tokenizer.word_counts) + 1

epochs = 1000
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
# model = TransformerXL(vocab_size, embed_dim, d_model=64, d_ff=ff_dim,

def cal_acc(real, pred):
        pred_labels = pred.T.argmax(axis = 1)
        acc = np.sum(np.equal(pred_labels, real))/len(real)

		# accuracies = tf.equal(real, tf.argmax(pred, axis=2))

		# mask = tf.math.logical_not(real == 0)
		# accuracies = tf.math.logical_and(mask, accuracies)

		# accuracies = tf.cast(accuracies, dtype=tf.float32)
		# mask = tf.cast(mask, dtype=tf.float32)
		return acc

class CustomLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        # TODO: Update document
        super(CustomLearningRate, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        
    def __call__(self, step_num):
        # TODO: Update document
        lrate = tf.cast(self.d_model, tf.float32) ** (-0.5) * tf.math.minimum(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5) )
        return lrate
model = TransformerXL(n_vocab=sentences_tokenizer_size,
                        d_embed=EMBEDDING_SIZE,
                        d_model=HIDDEN_SIZE,
                        d_ff=FFN_SIZE,
                        q_len=SEQ_LEN,
                        m_len=MEM_LEN,
                        num_heads=NUM_HEADS,
                        n_layer=N_LAYER,
                        dropout_rate=0.1,
                        untie_rel_bias=True)

learning_rate = CustomLearningRate(512)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
checkpoint = tf.train.Checkpoint(model=model, optimizer = optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, './checkpoints', max_to_keep=3)

def loss_function(labels, logits):
    """
    Compute the loss function
    """
    loss = tf.keras.losses.categorical_crossentropy(
        labels, logits, from_logits=True)
    loss = tf.reduce_mean(loss)
    return loss
            
def train_step(inputs, labels, optimizer, inputs_mem):
    with tf.GradientTape() as tape:
        # print(labels.shape)
        labels = tf.keras.utils.to_categorical(labels, 2)
        # print(labels.shape)
        logits, new_mems = model(inputs, inputs_mem, training=True)
        # print(logits.shape)
        x = tf.keras.layers.GlobalAveragePooling1D()(logits)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(20, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        logits = tf.keras.layers.Dense(2, activation="softmax")(x)
        loss = loss_function(labels, logits)
        
#compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    #update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)
    # train_accuracy.update_state(cal_acc(labels, logits))
    return loss, new_mems, logits
    
def fit():
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")
    mems=None
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        for (batch, (inputs, labels)) in enumerate(train_dataset):
            loss, mems, logits = train_step(inputs, labels, optimizer, mems)
            train_loss(loss)
            # train_accuracy(labels, logits)
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy '.format(epoch + 1, batch, train_loss.result()))
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix = checkpoint_manager.save_path)
            print('Saved checkpoint for epoch {}'.format(epoch + 1))

fit()   
