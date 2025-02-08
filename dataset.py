import numpy as np
import torch

class HoroskopesDataSet:
    def __init__(self, data_path, zodiac_borders = None, token_len = 1, seed_words_num = 3):
        #Input data
        with open(data_path, 'r') as file:
            self.raw_data = file.readlines()
        self.zodiac_borders = zodiac_borders
        self.token_len = token_len

        #Make map
        #'<soss>' - start of seed sentence, '<eoss>' - end of seed sentence, '_' - pad token
        #'<soos>' - start of output sentence, '<eoos>' - end of output sentence
        tokens = set([' ', '_', '<soss>', '<eoss>', '<soos>', '<eoos>'])
        for sentence in self.raw_data:
            words = sentence.strip().split(" ")
            for word in words:
                for i in range(0, len(word), self.token_len):
                    tokens.add(word[i : i + token_len])
        self.num_tokens = len(tokens)
        self.token_to_idx = {x : idx for idx, x in enumerate(sorted(tokens))}
        self.idx_to_token = {idx : x for idx, x in enumerate(sorted(tokens))}

        #Tokenise text
        self.processed_seed_data = []
        self.seed_tokens_lengths = []
        self.processed_output_data = []
        max_seed_length = 0
        max_output_length = 0
        for sentence in self.raw_data:
            words = sentence.strip().split(" ")
            tokenised_seed_words = [self.token_to_idx['<soss>']]
            tokenised_output_words = [self.token_to_idx['<soos>']]
            #Seed words
            for word in words[:seed_words_num]:
                for i in range(0, len(word), self.token_len):
                    ngramm = word[i : i + token_len]
                    tokenised_seed_words.append(self.token_to_idx[ngramm])
                tokenised_seed_words.append(self.token_to_idx[' '])
            #Remove excess ' '
            tokenised_seed_words.pop()
            #Add eos token
            tokenised_seed_words.append(self.token_to_idx['<eoss>'])
            #Remember lengths
            self.seed_tokens_lengths.append(len(tokenised_seed_words))
            #Add to pool
            self.processed_seed_data.append(tokenised_seed_words)
            #Refresh max
            max_seed_length = max(max_seed_length, len(tokenised_seed_words))

            for word in words[seed_words_num:]:
                for i in range(0, len(word), self.token_len):
                    ngramm = word[i : i + token_len]
                    tokenised_output_words.append(self.token_to_idx[ngramm])
                tokenised_output_words.append(self.token_to_idx[' '])
            #Remove excess ' '
            tokenised_output_words.pop()
            #Add eos token
            tokenised_output_words.append(self.token_to_idx['<eoos>'])
            #Add to pool
            self.processed_output_data.append(tokenised_output_words)
            #Refresh max
            max_output_length = max(max_output_length, len(tokenised_output_words))

        self.processed_seed_data = np.array(self.make_padding(self.processed_seed_data, max_seed_length))
        self.processed_output_data = np.array(self.make_padding(self.processed_output_data, max_output_length))
        self.seed_tokens_lengths = np.array(self.seed_tokens_lengths)
        self.mask_output_data = (self.processed_output_data != self.token_to_idx['_'])

    def make_padding(self, data, max_len):
        res = []
        for sentence in data:
            res.append(sentence + [self.token_to_idx['_']] * (max_len - len(sentence)))
        return res

    def tokenise_sentence(self, sentence):
        words = sentence.strip().split(" ")
        res = []
        for word in words:
            for i in range(0, len(word), self.token_len):
                    ngramm = word[i : i + self.token_len]
                    res.append(self.token_to_idx[ngramm])
            res.append(self.token_to_idx[' '])
        #Remove excess ' '
        res.pop()
        return res

    def untokenise_sentence(self, tokenised_sentence):
        return "".join(self.idx_to_token[idx] for idx in tokenised_sentence)   

    def gen_batch(self, batch_size = 256, tensor = False):
        idxs = np.random.choice(self.processed_seed_data.shape[0], batch_size, replace=False)
        if tensor:
            return (torch.tensor(self.processed_seed_data[idxs]),
                     torch.tensor(self.processed_output_data[idxs]),
                     torch.tensor(self.seed_tokens_lengths[idxs]),
                     torch.tensor(self.mask_output_data[idxs]))
        else:
            return (self.processed_seed_data[idxs], self.processed_output_data[idxs],
                     self.seed_tokens_lengths[idxs], self.mask_output_data[idxs])
    
    def __len__(self):
        return len(self.processed_output_data)