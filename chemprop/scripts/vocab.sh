#!/bin/bash

for file in $(ls data/*.csv); do
	python ../models/jtnn.py < $file > "${file%%.*}".vocab &
done
