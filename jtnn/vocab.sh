#!/bin/bash

for file in $(ls data/*.csv); do
	python jtnn.py < $file > "${file%%.*}".vocab &
done
