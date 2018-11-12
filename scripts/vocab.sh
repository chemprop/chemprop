#!/bin/bash

for file in $(ls ../data/*.csv); do
	python vocab.py --data_path < $file > --vocab_path "${file%%.*}".vocab &
done
