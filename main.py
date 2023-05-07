# MAIN
from matplotlib import pyplot as plt

from tp1.tp1 import tp1
from tp2.tp2 import tp2

if __name__ == '__main__':

    args_tp1 = {
        'r_min': -10,
        'r_max': 10,
        'count': 45,
        'set_div_condition': lambda x: (x[0] > 0 and x[1] > 0) or (x[0] < 0 and x[1] < 0),
        'bchk_condition': lambda bchk_set: [1 if x[0] > 0 else 0 for x in bchk_set],
        'test_sample_size': 0.2
    }
    #tp1(**args_tp1)

    args_tp2 = {
        'dataset_filename': "data/spam_train.csv"
    }
    tp2(**args_tp2)

    plt.show()
