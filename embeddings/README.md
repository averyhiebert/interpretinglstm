# Notes
This directory contains two sets of word embeddings produced from
character-level models, one derived from War and Peace and one from the
LOB corpus.  I didn't include any other data (e.g. recorded 
hidden states on which the vectors were based) since it's large, not that 
interesting, and can be generated using the scripts provided, 
but I can send further data by email if requested.

Embeddings are presented in text form (space-seperated values).  
Github does not allow files larger than 100MB, so I've
compressed the War and Peace file, and split the LOB file up into three pieces.
You can recover the original file using 
`cat LOBvectors1.txt LOBvectors2.txt LOBvectors3.txt > LOBvectors.txt`.

The `experiments.py` script loads vectors as pickled python objects, but
the pickle files were too large for github.  The pickle files in the
appropriate format can be generated using the `get_word_vecs` action of
the `experiments.py` script (with appropriate arguments, which I haven't
documented, but which I will document if anyone expresses interest in this),
which in trun requires using the script to generate the raw hidden state data,
which I also haven't documented but will help with if requested.

Email me if you have any questions.
