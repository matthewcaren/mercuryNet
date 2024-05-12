# mercury

A system for video-to-prosody synthesis.

The Roman god Mercury is the god of translators and interpreters.


## running train/test

Train: `python model/train_mercury.py -r [ROOT_DIR] -w [WINDOW_LOC] -c [DATA_COUNT] -b [BATCH_SIZE] -e [EPOCHS] -cp [CHECKPOINT PATH]`

Infer `python model/infer.py -r [ROOT_DIR] -w [WINDOW_LOC] -c [DATA_COUNT] -cp [CHECKPOINT_PATH]`
