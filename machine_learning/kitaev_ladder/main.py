import train_evaluate_ae as teea
import train_evaluate_fc as tefc
import diagrams as dgrm
import diagrams_new as dgrmnew
import pandas as pd

## TRAIN AND EVALUATE FC MODEL (PHYSICAL)
d = 32

tmfc = tefc.Training(
    model_save_to=f'./models/kitaev_ladder_fc_physical_horizontal_D{d}.pth',
    feature_length=14,
    model_type='fc'
)

tmfc.train_model(
    data_path=f'../../data_set/kitaev_ladder/fc/physical_horizontal_D{d}_train.csv',
    epoch_num=10,
    lr=0.01,
    batch_size=10,
    weight_decay=0.001
)

tmfc.evaluate_model(
    data_path=f'../../data_set/kitaev_ladder/fc/physical_D{d}_all.csv',
    predict_save_to=f'predict_fc_physical_horizontal_D{d}_all.csv'
)

## DIAGRAMS FOR FC TRAINING
d = 32
data = pd.read_csv(f'./predict/predict_fc_physical_D{d}.csv')
data = data.drop(columns=['hz']).rename(columns={'hx': 'x'}).sort_values('x')
dg = dgrm.Diagram(d)
dg.phases_probability(data, x_title='$h_x$')

data = pd.read_csv(f'./predict/predict_fc_D{d}_hz0.csv')
data = data.drop(columns=['hz']).rename(columns={'hx': 'x'}).sort_values('x')
dg = dgrm.Diagram(data, d)
dg.phases_probability(x_title='$h_z$')

## TRAIN AND EVALUATE FC MODEL (LAMBDA)
d = 32

tmfc = tefc.Training(
    model_save_to=f'./models/kitaev_ladder_fc_D{d}.pth',
    feature_length=d,
    model_type='fc'
)

tmfc.train_model(
    data_path=f'../../data_set/kitaev_ladder/lambda_D{d}_train.csv',
    epoch_num=30,
    lr=0.1,
    batch_size=10,
    weight_decay=0.01
)

tmfc.evaluate_model(
    data_path=f'../../data_set/kitaev_ladder/lambda_D{d}_test.csv',
    predict_save_to=f'predict_fc_D{d}.csv'
)

## DIAGRAMS FOR FC TRAINING
data = pd.read_csv(f'./predict/predict_fc_D{d}_hx0.csv')
data = data.drop(columns=['hx']).rename(columns={'hz': 'x'}).sort_values('x')
dg = dgrm.Diagram(data, d)
dg.phases_probability(x_title='$h_x$')

data = pd.read_csv(f'./predict/predict_fc_D{d}_hz0.csv')
data = data.drop(columns=['hz']).rename(columns={'hx': 'x'}).sort_values('x')
dg = dgrm.Diagram(data, d)
dg.phases_probability(x_title='$h_z$')


## TRAIN AND EVALUATE AE MODEL (LAMBDA NORMALIZED)
d = 32
tmea = teea.Training(
    model_save_to=f'./models/kitaev_ladder_ae_D{d}_lambda_normalized_toric.pth',
    feature_length=32
)

tmea.train_model(
    data_path=f'../../data_set/kitaev_ladder/ae/lambda_normalized_D{d}_toric_phase.csv',
    epoch_num=8,
    lr=0.00091,
    batch_size=10,
    weight_decay=0.0001
)

tmea.evaluate_model(
    data_path=f'../../data_set/kitaev_ladder/ae/lambda_normalized_D{d}_all.csv',
    predict_save_to=f'predict_ae_D{d}_lambda_normalized_toric.csv'
)

# DIAGRAMS FOR AE TRAINING
dg = dgrm.Diagram(d)
data = pd.read_csv(f'./predict/predict_ae_D{d}_lambda_normalized_toric.csv')
dg.phase_diagram_2d(data)
dg.gradient_test(data)
dg.countor_plot(data)
# dg.three_dim_plot(data)


## TRAIN AND EVALUATE AE MODEL (LAMBDA)
d = 32
tmea = teea.Training(
    model_save_to=f'./models/kitaev_ladder_ae_D{d}_labmda_toric.pth',
    feature_length=d
)

tmea.train_model(
    data_path=f'../../data_set/kitaev_ladder/ae/lambda_D{d}_toric_phase.csv',
    epoch_num=10,
    lr=0.001,
    batch_size=20,
    weight_decay=0.0001
)

tmea.evaluate_model(
    data_path=f'../../data_set/kitaev_ladder/ae/lambda_D{d}_all.csv',
    predict_save_to=f'predict_ae_D{d}_lambda_toric.csv'
)

## DIAGRAMS FOR AE TRAINING
dg = dgrm.Diagram(d)
# data = pd.read_csv(f'./predict/predict_ae_D{d}_lambda_toric.csv')
dg.phase_diagram_2d(
    pd.read_csv(f'./predict/predict_ae_D{d}_lambda_toric.csv')
)
# dg.gradient_test(data)
dg.vector_plot(
    pd.read_csv(f'./predict/predict_ae_D{d}_lambda_toric.csv'),
    pd.read_csv(f'./predict/predict_ae_D{d}_lambda_toric.csv'),
    pd.read_csv(f'./predict/predict_ae_D{d}_lambda_toric.csv'),
)
# dg.three_dim_plot(data)


## TRAIN AND EVALUATE AE MODEL (PHYSICAL PROFILE)
# d = 32
my_dict = {
    # # PHYSICAL NORMALIZED
    # 'toric': {
    #     'feature_len': 13,
    #     'model': './models/kitaev_ladder_ae_D32_physical_normalized_toric_best.pth',
    #     'train': '../../data_set/kitaev_ladder/ae/physical_profile_normalized_D32_toric_phase.csv',
    #     'predict': 'predict_ae_D32_physical_normalized_toric.csv',
    #     'eval_data': '../../data_set/kitaev_ladder/ae/physical_profile_normalized_D32_all.csv'
    # },
    # 'plzdx': {
    #     'feature_len': 13,
    #     'model': './models/kitaev_ladder_ae_D32_physical_normalized_plzd_x.pth',
    #     'train': '../../data_set/kitaev_ladder/ae/physical_profile_normalized_D32_plzd_x_phase.csv',
    #     'predict': 'predict_ae_D32_physical_normalized_plzd_x.csv',
    #     'eval_data': '../../data_set/kitaev_ladder/ae/physical_profile_normalized_D32_all.csv'
    # },
    # 'plzdz': {
    #     'feature_len': 13,
    #     'model': './models/kitaev_ladder_ae_D32_physical_normalized_plzd_z.pth',
    #     'train': '../../data_set/kitaev_ladder/ae/physical_profile_normalized_D32_plzd_z_phase.csv',
    #     'predict': 'predict_ae_D32_physical_normalized_plzd_z.csv',
    #     'eval_data': '../../data_set/kitaev_ladder/ae/physical_profile_normalized_D32_all.csv'
    # },
    # # LAMBDA
    'toric': {
        'feature_len': 32,
        'model': './models/kitaev_ladder_ae_D32_lambda_toric.pth',
        'train': '../../data_set/kitaev_ladder/ae/lambda_D32_toric_phase.csv',
        'predict': 'predict_ae_D32_lambda_toric.csv',
        'eval_data': '../../data_set/kitaev_ladder/ae/lambda_D32_all.csv'
    },
    'plzdx': {
        'feature_len': 32,
        'model': './models/kitaev_ladder_ae_D32_lambda_plzd_x.pth',
        'train': '../../data_set/kitaev_ladder/ae/lambda_D32_plzd_x_phase.csv',
        'predict': 'predict_ae_D32_lambda_plzd_x.csv',
        'eval_data': '../../data_set/kitaev_ladder/ae/lambda_D32_all.csv'
    },
    'plzdz': {
        'feature_len': 32,
        'model': './models/kitaev_ladder_ae_D32_lambda_plzd_z.pth',
        'train': '../../data_set/kitaev_ladder/ae/lambda_D32_plzd_z_phase.csv',
        'predict': 'predict_ae_D32_lambda_plzd_z.csv',
        'eval_data': '../../data_set/kitaev_ladder/ae/lambda_D32_all.csv'
    },
}

phase_name = 'toric'

tmea = teea.Training(
    model_save_to=my_dict[phase_name]['model'],
    feature_length=my_dict[phase_name]['feature_len']
)

tmea.train_model(
    data_path=my_dict[phase_name]['train'],
    epoch_num=1000,
    lr=0.00001,
    batch_size=5,
    weight_decay=0.000001
)

tmea.evaluate_model(
    data_path=my_dict[phase_name]['eval_data'],
    predict_save_to=my_dict[phase_name]['predict']
)

# DIAGRAMS FOR AE TRAINING
d = 32
dg = dgrm.Diagram(d)
data = pd.read_csv(f'./predict/predict_ae_D32_physical_normalized_toric.csv')
dg.phase_diagram_2d(data)
dg.gradient_test(data)
dg.countor_plot(data)
dg.vector_plot(
    pd.read_csv(f'./predict/predict_ae_D32_physical_normalized_toric.csv'),
    pd.read_csv(f'./predict/predict_ae_D32_physical_normalized_plzd_x.csv'),
    pd.read_csv(f'./predict/predict_ae_D32_physical_normalized_plzd_z.csv'),
)
dg.three_dim_plot()
# dg.three_dim_plot(data)
dg.three_dim_stream_plot(data)

dg.surface_inner_product()


d = 32
dg = dgrmnew.Diagram(d)
# dg.phase_transition_border_predict_plot('toric')
# dg.phase_transition_border_predict_plzd_plot()
# dg.ae_phase_diagram_toric()
dg.ae_stream_plot('toric', 'lambda')
# dg.mag_phase_diagram_toric()
# dg.chishod()


# dg = dgrm.Diagram(32)
# data = pd.read_csv(f'./predict/predict_ae_D32_lambda_toric.csv')
# dg.vector_plot(data)
# dg.energy_diagram(data)
