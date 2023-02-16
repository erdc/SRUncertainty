import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

path = os.getcwd()
parent = os.path.join(path, os.pardir)
parent2 = os.path.join(parent, os.pardir)
sys.path.append(os.path.join(parent2,'code'))

from networks import network
from plotters import plotter
from trainers import trainer
import settings
import data


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
tf.keras.backend.clear_session()

def get_inputs():
    cases = ['DUCT_1100',
        'DUCT_1150',
        'DUCT_1250',
        'DUCT_1300',
        'DUCT_1350',
        'DUCT_1400',
        'DUCT_1500',
        'DUCT_1600',
        'DUCT_1800',
        'DUCT_2000',
        'DUCT_2205',
        'DUCT_2400',
        'DUCT_2600',
        'DUCT_2900',
        'DUCT_3200',
        'DUCT_3500',
        'PHLL_case_0p5',
        'PHLL_case_0p8',
        'PHLL_case_1p0',
        'PHLL_case_1p2',
        'PHLL_case_1p5',
        'BUMP_h20',
        'BUMP_h26',
        'BUMP_h31',
        'BUMP_h38',
        'BUMP_h42',
        'CNDV_12600',
        'CNDV_20580',
        'CBFS_13700']

    variables = ['gradU', 'S', 'R', 'Shat', 'Rhat', 'gradk', 'gradp',
                 'Ak', 'Ap', 'Akhat', 'Aphat', 'T_t', 'T_k',
                 'Tensors', 'Lambda', 'I',
                 'q[:, 0]', 'q[:, 1]', 'q[:, 2]', 'q[:, 3]',
                 'wallDistance', 'DUDt']
    labels = ['tau', 'k', 'b']
    models = ['kepsilon', 'kepsilonphitf', 'komega', 'komegasst']
    print("Welcome to the Turbulence Dataset ML API. We will now iteratively go through the options, allowing you to select inputs.")
    print("Once you know the layout of these options, you can input them as arguments separated by a -")
    print("e.g. 'ann-1D-50-kepsilon-DUCT_1100 DUCT_1150-S R-DUCT_1100 DUCT_1150-S R'")
    print("Which ML model do you want to use? ['gan', 'unet']")
    ml_model = input("Enter: 'unet' or 'gan' to select ml model:\n")
    print("___________________________________________________________________________________")
    print("What RANS model type do you wish to use?")
    print("Possible models: ", models)
    model = input("Enter the model now (do not include the ''):\n")
    print("___________________________________________________________________________________")
    print("What case(s) to be trained on?")
    print("Possible cases: ", cases)
    training_case = input("Enter the case now, separated by a space if you want multiple (do not include the ''):\n")
    training_case = tuple(training_case.split(' '))
    print(f'You entered {training_case}')
    print("___________________________________________________________________________________")
    print("What variable(s) to be used for feature inputs?")
    print("Possible variables: ", variables)
    input_fields = input("Enter the variables now, separated by a space if you want multiple (do not include the ''):\n")
    input_fields = tuple(input_fields.split(' '))
    print(f'You entered {input_fields}')
    print("___________________________________________________________________________________")
    print("What case(s) to be tested on?")
    print("Possible cases: ", cases)
    test_case = input("Enter the case now, separated by a space if you want multiple (do not include the ''):\n")
    test_case = tuple(test_case.split(' '))
    print(f'You entered {test_case}')
    print("___________________________________________________________________________________")
    print("What variable(s) to be used for feature targets?")
    print("Possible variables: ", variables)
    print("Additionally, you can select from the following DNS/LES labels: ", labels)
    target_fields = input("Enter the variables now, separated by a space if you want multiple (do not include the ''):\n")
    target_fields = tuple(target_fields.split(' '))
    print(f'You entered {target_fields}')
    print("___________________________________________________________________________________")
    print("Train or Predict")
    ml_model = input("Enter: 'train' or 'predict' to select mode:\n")
    print("___________________________________________________________________________________")
    
    return ml_model, gridded, model, training_case, input_fields, test_case, target_fields, mode

if __name__ == "__main__":
    """use the following argument as an example to run the code appropriately for McConkey et al:
    
    python network.py 'unet-kepsilon-PHLL_case_0p5 PHLL_case_0p8 PHLL_case_1p0 PHLL_case_1p2 BUMP_h20 BUMP_h26 BUMP_h31 BUMP_h38 CNDV_12600-Ux Uy-PHLL_case_1p5 BUMP_h42 CNDV_20580-Ux Uy-train'"""

    if len(sys.argv) > 1:
        args = ''.join(sys.argv[1])
        args = tuple(args.split('-'))
        ml_model = args[0]
        rans_model = tuple(args[1].split(' '))[0]
        training_case = tuple(args[2].split(' '))
        input_fields = tuple(args[3].split(' '))
        test_case = tuple(args[4].split(' '))
        target_fields = tuple(args[5].split(' '))
        mode = args[6]
        args = [ml_model, rans_model, training_case, input_fields, test_case, target_fields, mode]
    else:
        args = get_inputs()
        ml_model, gridded, rans_model, training_case, input_fields, test_case, target_fields, mode = args
        
    TurbulenceDataset = data.TurbulenceDataset()

    error_list = []
    sensor_list = []
    dataset_list = []
    x_coord = []
    y_coord = []
    train_sensor_mask = []
    test_sensor_mask = []
    train_sensor_gridded = []
    test_sensor_gridded = []
    train_features = []
    test_features = []
    train_input_features = []
    train_target_features = []
    test_input_features = []
    test_target_features = []
    wall_train_features = []
    wall_test_features = []

    for no_sensors in settings.sensor_list:
        print(no_sensors)
        no_sensors = int(no_sensors)
        sensor_list.append(no_sensors)
        try:
            print("Trying to load exact cases/fields from already pre-processed files...")
            dataset = np.load(parent2 + '/data/npy/2D_' + str(training_case) + str(test_case) + str(input_fields) + str(target_fields) + str(no_sensors) + ".npy", allow_pickle=True)
        except:
            print("Correct combination of files not found, pre-processing requested cases and fields")
            dataset = TurbulenceDataset.load_gridded_dataset(no_sensors, rans_model, training_case, input_fields, test_case, target_fields)

        x_coord.append(dataset[0])
        y_coord.append(dataset[1])
        train_sensor_mask.append(dataset[2])
        test_sensor_mask.append(dataset[3])
        train_sensor_gridded.append(dataset[4])
        test_sensor_gridded.append(dataset[5])
        train_features.append(dataset[6])
        test_features.append(dataset[7])
        train_input_features.append(dataset[8])
        train_target_features.append(dataset[9])
        test_input_features.append(dataset[10])
        test_target_features.append(dataset[11])
        wall_train_features.append(dataset[12])
        wall_test_features.append(dataset[13])
        
    data_list = [x_coord, y_coord, train_sensor_mask, test_sensor_mask, train_sensor_gridded, 
                 test_sensor_gridded, train_features, test_features, train_input_features, train_target_features,
                 test_input_features, test_target_features, wall_train_features, wall_test_features]
    
    settings.input_features = np.asarray(train_input_features[0]).shape[-1] + 1
    settings.target_features = np.asarray(train_target_features[0]).shape[-1]

    model = network((settings.input_rows, settings.input_cols, settings.input_features), (settings.target_rows, settings.target_cols, settings.target_features), arch=ml_model, gaussian=False, noise=0, avg=False)
    
    model = model.get(path = path, args=args)
    
    if ml_model == 'unet':

        model.compile(optimizer=tf.keras.optimizers.Nadam(), loss="mean_squared_error", metrics=['mae', 'mse'])

    elif ml_model == 'gan':
        model.compile(d_optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4, beta_1=0.5),
                      g_optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4, beta_1=0.5),
                      loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True))        #TurbuleNet.train_and_evaluate_unet()
        
    if mode == 'train':
        
        error = trainer().train_and_evaluate_mf(ml_model, model, data_list, args)

    elif mode == 'predict':
        
        error = plotter().predict_and_plot_mf(ml_model, model, data_list, args)
 


