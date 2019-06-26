# blstm_crf_seg
Chinese (Tibetan) segmentation using BLSTM + CRF

Requiements:
    python == 3.0
    torch >= 0.4
    pickle
    tensorflow(optional, for tensorboard logging)

Using:
    Prepare data: train data, valid data, test data should be placed in the "data/" directory
    Process data: using data.py
    
    Train model: using train.py

    Test model: using test.py
