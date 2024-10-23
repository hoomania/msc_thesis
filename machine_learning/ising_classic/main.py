import numpy as np

import train_evaluate_fc as tefc
import train_evaluate_cnn as tecnn
import diagrams as dgrm

ll = 20

## TRAIN AND EVALUATE FC MODEL
tmfc = tefc.Training(
    model_save_to=f'./models/ising_classic_fc_L{ll}.pth',
    feature_length=ll
)

tmfc.train_model(
    data_path=f'../../data_set/ising_classic/train/data_square_L{ll}_train.csv',
    epoch_num=10,
    lr=0.01,
    batch_size=10,
    weight_decay=0.001
)

tmfc.evaluate_model(
    data_path=f'../../data_set/ising_classic/test/data_square_L{ll}_test.csv',
    predict_save_to=f'predict_fc_L{ll}.csv'
)

## DIAGRAMS FOR FC TRAINING
ll = 10
dg = dgrm.Diagram(f'./predict/predict_fc_L{ll}.csv', ll)
dg.phase_probability_boxplot('phase_p')
dg.phase_probability_boxplot('phase_f')
dg.phases_probability()



## TRAIN AND EVALUATE CNN MODEL
ll = 30
tmcnn = tecnn.Training(
    model_save_to=f'./models/ising_classic_cnn_L{ll}.pth',
    lattice_length=ll
)

tmcnn.train_model(
    images_path=f'../../data_set/ising_classic/cnn/train/L{ll}',
    annotation_path=f'../../data_set/ising_classic/cnn/train/annotation/cnn_L{ll}_train_annotation.csv',
    epoch_num=10,
    lr=0.001,
    batch_size=60,
    momentum=0.9
)

tmcnn.evaluate_model(
    images_path=f'../../data_set/ising_classic/cnn/test/L{ll}',
    annotation_path=f'../../data_set/ising_classic/cnn/test/annotation/cnn_L{ll}_test_annotation.csv',
    predict_save_to=f'predict_cnn_L{ll}.csv'
)

## DIAGRAMS FOR CNN TRAINING
# ll = 60
for ll in np.arange(10, 70, 10):
    dg = dgrm.Diagram(f'./predict/predict_cnn_L{ll}.csv', ll)
    dg.phase_probability_boxplot('phase_f')
    dg.phase_probability_boxplot('phase_p')
    dg.phases_probability()
