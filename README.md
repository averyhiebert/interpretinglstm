# Interpreting Word-Level Hidden State Behaviour of Character-Level LSTM Language Models

This repository contains code associated with the paper 
"Interpreting Word-Level Hidden State Behaviour of Character-Level LSTM Language Models", including two trained models (located in the `saved_models` 
directory) and the word embeddings derived from these models (in the
`embeddings` directory), as well as the scripts used for generating the
word embeddings and performing clustering of hidden states and other
experiments (`experiments.py`).

I apologize for the messiness of the code and the poor usability.
Blame me (Avery Hiebert) and not my co-authors. 
I found that the desire to refactor everything to make things presentable was
keeping me from actually getting around to publishing the code, so I decided to
just release the code more-or-less as-is.  I might try to add
documentation in the future if people are actually
interested.  I can also answer questions directed to averyhiebert@gmail.com.

## Copyright

All code in this repository is licensed under the MIT license (`LICENSE.txt`).
However, the included training texts (*War and Peace* and the LOB corpus) are
distributed under different licenses (see `training_data/copyright_info/README.md`).
