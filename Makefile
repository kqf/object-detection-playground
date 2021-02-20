competition = vinbigdata-chest-xray-abnormalities-detection
logdir = $(TENSORBOARD_DIR)/$(message)

develop: data/train
	python models/main.py --fin $^ --logdir=$(logdir)


all: weights/fold0.pt \
	 weights/fold1.pt \
	 weights/fold2.pt \
	 weights/fold3.pt \
	 weights/fold4.pt

weights/fold%.pt: foldname = $(basename $(@F))
weights/fold%.pt: logfold = $(logdir)-$(foldname)
weights/fold%.pt: data/train/fold%.json
	python models/train.py --fin $< --logdir $(logfold)
	gsutil -m cp thresholds.png $(logfold)
	gsutil -m cp $(logfold)/train_end_params.pt $@

data/train/fold%.json: data/train
	python models/split.py --fin $^ --fout $(@D)


infer:
	python models/infer.py


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
	cp -R models .tmp_submit/models
	cp -R weights .tmp_submit/weights
	rm .tmp_submit/models/kernel-metadata.json
	kaggle datasets version -p .tmp_submit -r zip -m "$(message)"
	rm -rf .tmp_submit


push-kernels:
	kaggle kernels push -p models/


.PHONY: infer develop push-data push-kernels
.PRECIOUS: data/train/fold%.json
