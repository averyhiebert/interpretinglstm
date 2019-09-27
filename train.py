''' Partially based on 
https://github.com/pytorch/examples/blob/master/word_language_model/main.py

There's a bunch of stuff hard coded here that really shouldn't be hard coded.
Sorry.'''

from datetime import datetime
import argparse

import matplotlib.pyplot as plt
import torch.optim as optim

from charlstm import *

torch.manual_seed(1)

# Set up for possibly using CUDA
USE_CUDA = True
device = torch.device("cuda" if USE_CUDA else "cpu")

# Some constants (may become arguments later)
# Training options
BATCH_SIZE = 10
EPOCHS = 3
BPTT = 200
LR = 0.01 # Learning rate
M = 0.99 # Momentum
DROPOUT = 0.5 # Dropout
SAVE_INTERVAL = 5 # How often to save the latest model (overwriting previous)

# Data Preprocessing ==========================================================
def load_data(filename):
    ''' Return the "vocab" of characters in the file, and tensors containing
    the train & test data as integer indices.'''
    with open(filename,"r") as f:
        file_content = f.read()
    
    charset = list(set(file_content))
    char2index = {c:i for i, c in enumerate(charset)}
    data = [char2index[c] for c in file_content]
    # === Split into 95% training data, 5% validation data:
    cutoff = (95*len(data))//100
    train_data = torch.LongTensor(data[0:cutoff])
    val_data = torch.LongTensor(data[cutoff:])

    # === Alternately, take the validation data from the beginning rather than
    #  from the end of the text:
    #cutoff = (5*len(data))//100
    #val_data = torch.LongTensor(data[0:cutoff])
    #train_data = torch.LongTensor(data[cutoff:])

    # Note: We trained our War and Peace model taking the validation data from
    #  the end of the book instead of the beginning, but this is actually not
    #  ideal since the final epilogue has some structural differences from the
    #  rest of the book.  You will get slightly better results if you take
    #  validation data from the start of the text instead.
    return charset, char2index, train_data, val_data


def batchify(data, batch_size):
    ''' Format a list of data points into batches,
    assuming data is currently a sequence of char indexes. '''
    ''' Each batch is a column. '''
    num_batches = data.size(0)//batch_size
    data = data.narrow(0,0, num_batches*batch_size)
    data = data.view(batch_size,-1).t().contiguous()
    return data.to(device)

def detach_history(h):
    ''' For truncated BPTT '''
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detach_history(v) for v in h)

def get_batch(data_source, i):
    ''' Should pass in already "batchified" data '''
    length = min(BPTT, len(data_source) - 1 - i)
    data = data_source[i:i+length]
    target = data_source[i+1:i+1+length].view(-1)
    return data, target

# Other tools ===========================================================

def evaluate(model, data_source, criterion):
    ''' "data_source" is already "batchified" '''
    batch_size = data_source.size(1)
    n_chars = len(model.charset)
    model.eval()
    loss = 0
    h, c = model.init_hidden(batch_size)
    hidden = (h.to(device), c.to(device))
    with torch.no_grad():
        for i in range(0, data_source.size(0) -1, BPTT):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1,n_chars)
            loss += len(data) * criterion(output_flat, targets).item() 
            hidden = detach_history(hidden) # Is this even necessary?
    return loss / len(data_source)

def save_latest(model, train_losses, val_losses):
    model.save_with_info("saved_models/latest")
    data = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open("saved_models/latest/loss_data","w") as f:
        f.write(json.dumps(data))

# Training ===================================================================

def train(model, train_data, criterion, optimizer, log_interval=100):
    ''' train_data is already "batchified" '''
    batch_size = train_data.size(1)
    model.train()
    total_loss = 0
    cur_loss = 0
    start_time = time.time()

    n_chars = len(model.charset)
    h, c = model.init_hidden(batch_size)
    hidden = (h.to(device), c.to(device))
    for batch, i in enumerate(range(0, train_data.size(0)-1, BPTT)):
        data, targets = get_batch(train_data, i)
        #print(data.size())  # DEBUG
        hidden = detach_history(hidden) # for truncated BPTT
        model.zero_grad()
        optimizer.zero_grad()

        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1,n_chars), targets)
        loss.backward()
        optimizer.step()


        total_loss += float(loss.item())

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            total_loss = 0
            elapsed = time.time() - start_time
            print("Current loss: %.5f" % cur_loss)
            print("ms per batch: %.3f" % (elapsed*1000/log_interval))
            start_time = time.time()

    return cur_loss

# Actually start training =====================================================

# Load data
batch_size = BATCH_SIZE
charset, char2index, train_data, val_data = load_data("training_data/WaP.txt")
#charset, char2index, train_data, val_data = load_data("training_data/LOB_nomarkup.txt")
train_data = batchify(train_data,batch_size)
val_data = batchify(val_data,batch_size)

# Create the model
model = CharLSTM(charset, 256, 512, 3, dropout=DROPOUT)
model.to(device)
model.set_device(device)
lr = LR
m = M

# Track training & validation losses for plotting & lr decay
train_losses = []
val_losses = []
def plot_losses():
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title("Loss per Epoch")
    plt.legend(["training","validation"], loc="upper left")
    plt.show()

# Create optimizer & criterion
criterion = nn.NLLLoss()
# Settings I used when training the "main" WaP model:
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=m)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    mode='min',factor=0.5,patience=10,verbose=True,cooldown=1)

# Settings I used when training the LOB model:
#optimizer = optim.Adagrad(model.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)

# (Again, I'm sorry that this is hard-coded etc.)


try:
    for epoch in range(1, EPOCHS + 1):
        print("="*80)
        epoch_start_time = time.time()
        new_tl = train(model, train_data, criterion, optimizer, log_interval=60)
        val_loss = evaluate(model,val_data,criterion)

        train_losses.append(new_tl)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print("\nEnd of epoch %d" % epoch)
        print("Elapsed time: %.3fs" % (time.time() - epoch_start_time))
        print("Validation loss: %.5f" % val_loss)
        print("Sample text:\n" + model.gen_text("\n",400))
        
        if epoch % SAVE_INTERVAL == 0:
            save_latest(model,train_losses,val_losses)
            print("Saved latest model")

        print("="*80)
except KeyboardInterrupt:
    print("Stopping training.")

