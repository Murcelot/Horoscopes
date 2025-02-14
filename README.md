# horoscopes
Model to generate horoscopes


Dataset was found on different sites and parced by me. It consists of 365 horoscopes for every zodiac sign, so there are 4380 different horoscopes. Each horoscope was split in sentences, each sentence was split on seed part (3-5 words) and target part (remain part). I thought about generating horoscopes based on zodiac sign, i. e. different horoscopes for different signs, but when i analyse dataset i found that there isn't any dependencies on zodiac sign))).

In dataset.py there is class that reads dataset, makes ngramm-token map, splits data in seed part and target part (for easy learning in encoder-decoder models) and generates batches. Also class helps in 'tokenise' and 'untokenise' routine for futher application.

As model i used RNN(GRU) with attention mechanism in seed part. Bidirectional GRU computes hiddens using seed part and then other GRU using previus hidden and seed hiddens predict next token. As Loss function i choosed MaskedNLLL. Crossentropy don't work properly, possibly due to big amount of pad tokens in target sentences.

In models.py you can find Encoder which perfoms computing hiddens from seed part, Attn module providing different types of attention mechanism and Decoder, which performs end prediction.
In train.py - training loop performs train with teacher forcing (previous token is actual token) or without it (previous token is previous decoder's output).
In eval_module.py - you can find Generator class. This class provides easy application of a model saved in file (that train.py return). Besides this generator has 2 modes: 'base' and 'beam_search'. Beam search choosing part of generated sentences with most probability among all tokens in part of sentences (not only the next token like in base mode). You can choose 'depth' and 'topk' parameters. 'topk' is a number of tokens with greater proba (which then will be used as start of part of sentence) and 'depth' is a length of that part. 
