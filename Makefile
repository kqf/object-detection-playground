competition = vinbigdata-chest-xray-abnormalities-detection
logdir = $(TENSORBOARD_DIR)/$(message)

develop: data/train/processed
# 	python scripts/main.py --fin $^ --logdir=$(logdir)
	echo "Training"

data/train/processed: data/train
	python scripts/preprocess.py --fin $^ --fout $@

	# Remove the raw files the directory 
	rm $^/*.dicom

all: weights/fold0.pt \
	 weights/fold1.pt \
	 weights/fold2.pt \
	 weights/fold3.pt \
	 weights/fold4.pt

weights/fold%.pt: foldname = $(basename $(@F))
weights/fold%.pt: logfold = $(logdir)-$(foldname)
weights/fold%.pt: data/train/fold%.json
	python detection/train.py --fin $< --logdir $(logfold)
	gsutil -m cp thresholds.png $(logfold)
	gsutil -m cp $(logfold)/train_end_params.pt $@

data/train/fold%.json: data/train
	python detection/split.py --fin $^ --fout $(@D)


infer:
	python detection/infer.py


data/:
	mkdir -p $@
	kaggle competitions download -c $(competition) -p data
	unzip -qq data/$(competition).zip -d data/
	rm -rf data/$(competition).zip


push-data:
	mkdir -p .tmp_submit
	cp dataset-metadata.json .tmp_submit/
	cp requirements.txt .tmp_submit/
	cp setup.py .tmp_submit
	cp -R detection .tmp_submit/detection
	cp -R weights .tmp_submit/weights
	rm .tmp_submit/detection/kernel-metadata.json
	kaggle datasets version -p .tmp_submit -r zip -m "$(message)"
	rm -rf .tmp_submit


push-kernels:
	kaggle kernels push -p detection/


.PHONY: infer develop push-data push-kernels
.PRECIOUS: data/train/fold%.json
