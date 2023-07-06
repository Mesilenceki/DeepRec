#!/bin/bash

#use_feature_column
python -m tensorflow.distribute.launch python train.py  --steps 10000  --use_feature_columns

#embedding_variable
python -m tensorflow.distribute.launch python train.py  --steps 10000