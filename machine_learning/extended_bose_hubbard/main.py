import train_evaluate_fc as tefc
import train_evaluate_ae as teae
import diagrams as dgrm


## TRAIN AND EVALUATE FC MODEL
d = 16

tmfc = tefc.Training(
    model_save_to=f'./models/ebh_D{d}.pth',
    feature_length=32,
    virtual_dim=16,
    hidden_layer_nodes=64
)

tmfc.train_model(
    data_path=f'../../data_set/extended_bose_hubbard/fc/lambda_D16_train.csv',
    epoch_num=100,
    lr=0.01,
    batch_size=10,
    weight_decay=0.001
)

tmfc.evaluate_model(
    data_path=f'../../data_set/extended_bose_hubbard/fc/lambda_D16_test.csv',
    predict_save_to=f'predict_fc_D{d}.csv'
)

## DIAGRAMS FOR FC TRAINING
# dg = dgrm.Diagram(f'./predict/predict_fc_D16.csv', 16)
# dg.phase_probability_boxplot('phase_p')
# dg.phases_probability()


## TRAIN AND EVALUATE AE MODEL
my_dict = {
    # # PHYSICAL NORMALIZED
    # 'mi': {
    #     'feature_len': 22,
    #     'model': './models/ebh_ae_D16_physical_mi.pth',
    #     'train': '../../data_set/extended_bose_hubbard/ae/physical_D16_mi_phase.csv',
    #     'predict': 'predict_ae_D16_physical_mi.csv',
    #     'eval_data': '../../data_set/extended_bose_hubbard/ae/physical_D16_all.csv'
    # },
    # 'sf': {
    #     'feature_len': 22,
    #     'model': './models/ebh_ae_D16_physical_sf.pth',
    #     'train': '../../data_set/extended_bose_hubbard/ae/physical_D16_sf_phase.csv',
    #     'predict': 'predict_ae_D16_physical_sf.csv',
    #     'eval_data': '../../data_set/extended_bose_hubbard/ae/physical_D16_all.csv'
    # },
    # 'hi': {
    #     'feature_len': 22,
    #     'model': './models/ebh_ae_D16_physical_hi.pth',
    #     'train': '../../data_set/extended_bose_hubbard/ae/physical_D16_hi_phase.csv',
    #     'predict': 'predict_ae_D16_physical_hi.csv',
    #     'eval_data': '../../data_set/extended_bose_hubbard/ae/physical_D16_all.csv'
    # },
    # 'dw': {
    #     'feature_len': 22,
    #     'model': './models/ebh_ae_D16_physical_dw.pth',
    #     'train': '../../data_set/extended_bose_hubbard/ae/physical_D16_dw_phase.csv',
    #     'predict': 'predict_ae_D16_physical_dw.csv',
    #     'eval_data': '../../data_set/extended_bose_hubbard/ae/physical_D16_all.csv'
    # },
    # # LAMBDA
    'mi': {
        'feature_len': 32,
        'model': './models/ebh_ae_D16_lambda_mi.pth',
        'train': '../../data_set/extended_bose_hubbard/ae/lambda_D16_mi_phase.csv',
        'predict': 'predict_ae_D16_lambda_mi.csv',
        'eval_data': '../../data_set/extended_bose_hubbard/ae/lambda_D16_all.csv'
    },
    'sf': {
        'feature_len': 32,
        'model': './models/ebh_ae_D16_lambda_sf.pth',
        'train': '../../data_set/extended_bose_hubbard/ae/lambda_D16_sf_phase.csv',
        'predict': 'predict_ae_D16_lambda_sf.csv',
        'eval_data': '../../data_set/extended_bose_hubbard/ae/lambda_D16_all.csv'
    },
    'hi': {
        'feature_len': 32,
        'model': './models/ebh_ae_D16_lambda_hi.pth',
        'train': '../../data_set/extended_bose_hubbard/ae/lambda_D16_hi_phase.csv',
        'predict': 'predict_ae_D16_lambda_hi.csv',
        'eval_data': '../../data_set/extended_bose_hubbard/ae/lambda_D16_all.csv'
    },
    'dw': {
        'feature_len': 32,
        'model': './models/ebh_ae_D16_lambda_dw.pth',
        'train': '../../data_set/extended_bose_hubbard/ae/lambda_D16_dw_phase.csv',
        'predict': 'predict_ae_D16_lambda_dw.csv',
        'eval_data': '../../data_set/extended_bose_hubbard/ae/lambda_D16_all.csv'
    },
}

phase_name = 'dw'
tmfc = teae.Training(
    model_save_to=my_dict[phase_name]['model'],
    feature_length=my_dict[phase_name]['feature_len']
)

tmfc.train_model(
    data_path=my_dict[phase_name]['train'],
    epoch_num=600,
    lr=0.001,
    batch_size=10,
    weight_decay=0.0001
)

tmfc.evaluate_model(
    data_path=my_dict[phase_name]['eval_data'],
    predict_save_to=my_dict[phase_name]['predict']
)

## DIAGRAMS FOR FC TRAINING
dg = dgrm.Diagram()
dg.ae_phase_diagram(phase_name, 'lambda')
dg.ae_stream_diagram(phase_name, 'lambda')
dg.four_phase_line_plot(phase_name, 'lambda')
dg.mu_secret()
