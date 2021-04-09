competition = vinbigdata-chest-xray-abnormalities-detection
logdir = $(TENSORBOARD_DIR)/$(message)

develop: data/train/processed
# 	python scripts/main.py --fin $^ --logdir=$(logdir)
	echo "Training"

data/train/processed: data/train
	python scripts/preprocess.py --fin $^ --fout $@

	# Remove the raw files the directory 
	rm $^/*.dicom


.PHONY: infer develop push-data push-kernels
.PRECIOUS: data/train/fold%.json
