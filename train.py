from torch import nn
import torch
from torch.optim import Adam

import random
from tqdm import tqdm

from models import EncoderRNN, LuongAttnDecoderRNN
from dataset import HoroskopesDataSet

#Masked is needed to avoid padding in target sentence
#Define masked negative log likelihood loss
def maskNLLLoss(inp, target, mask):
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1,1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return torch.tensor(0) if loss.isnan() else loss

#Define masked accuracy
def comp_accuracy(preds, target, mask):
    masked_preds = preds.masked_select(mask)
    masked_target = target.masked_select(mask)
    res = (masked_preds == masked_target).sum() / mask.sum()
    return torch.tensor(0) if res.isnan() else res

#Perform one batch step training
def train(input_variable, target_variable, lengths, batch_mask, encoder, decoder,
          encoder_optimizer, decoder_optimizer, clip, teacher_forcing_ratio):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    batch_mask = batch_mask.to(device)
    # Lengths for RNN packing should always be on the CPU
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0.
    accuracy = 0.
    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder inputs
    decoder_input = target_variable[:, 0].unsqueeze(1)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(1, target_variable.shape[1]):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #Predictions
            _, preds = decoder_output.topk(1)
            # Teacher forcing: next input is current target
            decoder_input = target_variable[:, t].unsqueeze(1)
            # Calculate loss and metrics            
            one_loss = maskNLLLoss(decoder_output, target_variable[:, t], batch_mask[:, t])
            one_accuracy = comp_accuracy(preds.squeeze(), target_variable[:, t], batch_mask[:, t])
            accuracy += one_accuracy
            loss += one_loss

    else:
        for t in range(1, target_variable.shape[1]):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # No teacher forcing: next input is decoder's own current output
            _, preds = decoder_output.topk(1)
            decoder_input = preds.to(device)
            #Remember logits and compute metrics            
            one_loss = maskNLLLoss(decoder_output, target_variable[:, t], batch_mask[:, t])
            one_accuracy = comp_accuracy(preds.squeeze(), target_variable[:, t], batch_mask[:, t])
            accuracy += one_accuracy
            loss += one_loss

    #Compute loss in batch sentences (targets shifted)
    accuracy /= (target_variable.shape[1] - 1)
    # Perform backpropagation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), accuracy.item()

def trainloop(encoder, decoder, num_epochs, batch_size, iters_per_epoch):
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder_opt = Adam(encoder.parameters(), lr = learning_rate)
    decoder_opt = Adam(decoder.parameters(), lr = learning_rate / teacher_forcing_ratio)

    # Training and validating loop
    for epoch in range(num_epochs):
        epoch_loss = 0.
        epoch_acc = 0.
        print('Epoch: {}/{}'.format(epoch, num_epochs - 1))
        print("Training...")
        encoder.train()
        decoder.train()
        for i in enumerate(tqdm(range(iters_per_epoch))):
            batch_input, batch_target, batch_lengths, batch_mask = data.gen_batch(batch_size, True)
            batch_loss, batch_acc = train(batch_input, batch_target, batch_lengths, batch_mask, encoder, decoder, encoder_opt,
                          decoder_opt, clip, teacher_forcing_ratio)
            epoch_loss += batch_loss
            epoch_acc += batch_acc
        epoch_loss /= iters_per_epoch
        epoch_acc /= iters_per_epoch
        print('Epoch loss: {:.3f}, epoch accuracy {:.2f}'.format(epoch_loss, epoch_acc))
        print('Validating...')
        encoder.eval()
        decoder.eval() 
        val_input, val_target, val_lengths, _ = data.gen_batch(3)
        #Print trues and preds
        for i in range(3):
            print('Truth: ' + data.untokenise_sentence(val_input[i]).strip('_') + 
                  data.untokenise_sentence(val_target[i]).strip('_'))
            print('Pred: ' + data.untokenise_sentence(val_input[i]).strip('_') + 
                  evaluate_model(encoder, decoder, torch.tensor(val_input[i]), torch.tensor(val_lengths[i])))
    return 

def evaluate_model(encoder, decoder, seed, seed_length, max_length = 50):  
    seed = seed.unsqueeze(0).to(device)
    seed_length = seed_length.unsqueeze(0).to('cpu')

    encoder_outputs, encoder_hidden = encoder(seed, seed_length)
    decoder_input = torch.tensor([[data.token_to_idx['<soos>']]]).to(device)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    res = [data.token_to_idx['<soos>']]
    for t in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_hidden = decoder_hidden.squeeze(1)
        #Predictions 
        _, next_token = decoder_output.topk(1)
        #Recursive
        decoder_input = next_token
        #Remember tokens
        res.append(next_token.item())
        if next_token.item() == data.token_to_idx['<eoos>']:
            break
    return data.untokenise_sentence(res)

#Using cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Data params
token_len = 2
num_seed_words = 3
data = HoroskopesDataSet('/home/makasin/Projects/Goroskops/horoskopes_wthout_shame/data/horoskopes_sentences.txt', token_len=token_len, seed_words_num=num_seed_words)

#Models params
num_tokens = data.num_tokens
embedding_size = 256
hidden_size = 256
n_layers = 1
attn_model = 'dot'
clip = 50
teacher_forcing_ratio = 0.9
learning_rate = 0.001
embedding = nn.Embedding(num_tokens, embedding_size)
encoder = EncoderRNN(hidden_size, embedding, embedding_size, n_layers)
decoder = LuongAttnDecoderRNN(attn_model, num_tokens, embedding, embedding_size, hidden_size, n_layers)

print('Number of tokens: ', data.num_tokens)
print('Input [num_epochs] [batch_size] [iters_per_epoch]')
num_epochs, batch_size, iters_per_epoch = map(int, input().split())

print('Input filename to save model and dict (in the same directory)')
filename = input()

#Performe training
trainloop(encoder, decoder, num_epochs, batch_size, iters_per_epoch)

#Saving results
torch.save({'emb' : embedding,
            'enc' : encoder,
            'dec' : decoder,
            'dict_token_idx' : data.token_to_idx,
            'dict_idx_token' : data.idx_to_token,
            'token_len' : token_len,
            'num_seed_words' : num_seed_words}, "." + filename)

#Done!

