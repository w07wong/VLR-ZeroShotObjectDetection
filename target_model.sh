#! /bin/bash

for i in `ls ../YCB_Video_Models/models/`;
do
	mkdir -p ../YCB_Video_Models/target/${i}/
	python data_gen.py --model-dir ../YCB_Video_Models/models/${i} --output-dir ../YCB_Video_Models/target/${i}/ --image-size 480 640
done
