import torch
import torch.nn.functional as F
import numpy as np

class Generator:
    def __init__(self, model_path, mode = 'base'):
        model_params = torch.load(model_path, map_location=torch.device('cpu'))
        self.token_len = model_params['token_len']
        self.num_seed_words = model_params['num_seed_words']
        self.encoder = model_params['enc']
        self.decoder = model_params['dec']
        self.token_to_idx = model_params['dict_token_idx']
        self.idx_to_token = model_params['dict_idx_token']
        self.num_tokens = len(self.token_to_idx)
        self.mode = mode
        self.tokens = torch.tensor([self.token_to_idx[tkn] for tkn in
                                         ['<soss>', '<eoss>', '<soos>', '<eoos>']]).view(4,1)

    def tokenise_sentence(self, sentence):
        words = sentence.lower().strip().split(" ")
        res = []
        for word in words:
            for i in range(0, len(word), self.token_len):
                    ngramm = word[i : i + self.token_len]
                    res.append(self.token_to_idx[ngramm])
            res.append(self.token_to_idx[' '])
        #Remove excess ' '
        res.pop()
        return torch.tensor(res), len(res)

    def untokenise_sentence(self, tokenised_sentence):
        return "".join(self.idx_to_token[idx] for idx in tokenised_sentence)   
    
    def generate_sentence(self, seed, max_length, topk = 3, depth = 5):
        if self.mode == 'base':
            seed, seed_length = self.tokenise_sentence(seed)
            seed = torch.cat([self.tokens[0], seed, self.tokens[1]]).unsqueeze(0)
            seed_length += 2
            seed_length = torch.tensor(seed_length).unsqueeze(0)
            encoder_outputs, encoder_hidden = self.encoder(seed, seed_length)
            
            decoder_input = self.tokens[2].view(1,1)
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]
            res = self.tokens[2].view(1)
            for t in range(max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_hidden = decoder_hidden.squeeze(1)
                #Predictions 
                _, next_token = decoder_output.topk(1)
                #Recursive
                decoder_input = next_token
                #Remember tokens
                res = torch.cat((res,next_token), dim = 0)
                if next_token.item() == self.tokens[3]:
                    break
            #Return generated sentence without start and end tokens
        if self.mode == 'beam_search':
            seed, seed_length = self.tokenise_sentence(seed)
            seed = torch.cat([self.tokens[0], seed, self.tokens[1]]).unsqueeze(0)
            seed_length += 2
            seed_length = torch.tensor(seed_length).unsqueeze(0)
            encoder_outputs, encoder_hidden = self.encoder(seed, seed_length)
            
            decoder_input = self.tokens[2].view(1,1)
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]
            res = self.tokens[2].view(1)
            for t in range(max_length // depth):
                _, curr_sentence, decoder_hidden = self.beam_search(
                    encoder_outputs, decoder_input, decoder_hidden, topk, depth)
                decoder_input = curr_sentence[-1].view(1,1)
                res = torch.cat((res, curr_sentence))
                if decoder_input.item() == self.tokens[3]:
                    break
        return self.untokenise_sentence(res.data.numpy()[1:-1]) 

    def beam_search(self, encoder_outputs, decoder_input, decoder_hidden, topk = 3, depth = 3):
        #One step of searching
        def gen_next_token(curr_decoder_output, curr_decoder_hidden, encoder_outputs, d, score, sentence):
            #Stop in end of depth
            if d == 0:
                return score, sentence, curr_decoder_hidden
            #Stop if found end token and maximise score of this sentence
            if curr_decoder_output.item() == self.tokens[3]:
                return score, sentence, curr_decoder_hidden
            #Next output
            decoder_output, decoder_hidden = self.decoder(curr_decoder_output, curr_decoder_hidden, encoder_outputs)
            proba, next_token = decoder_output.topk(1)
            #Recursion
            return gen_next_token(next_token.view(1,1), decoder_hidden,
                                    encoder_outputs, d - 1, score * proba.item(), torch.cat((sentence, next_token)))

        decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
        #Chossing topk tokens
        probs, next_tokens = decoder_output.topk(topk)
        #Starts of sentences
        sentences = next_tokens.view(-1,1)
        results = [gen_next_token(sentences[i].view(1,1), decoder_hidden,
                        encoder_outputs, depth, probs[i].item(), sentences[i]) for i in range(topk)]
        #Return sentences with the greatest proba
        return max(results)