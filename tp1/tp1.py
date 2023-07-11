# TP1 Main

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from random_custom import RandomCustom as Rand


def gen_coords(**gen_args):
    r = Rand()

    set_x = r.gen_rand(
        np.array,
        lambda r_min, r_max, count: [random.randrange(r_min, r_max, 1) for i in range(count)],
        **gen_args
    )

    set_y = r.gen_rand(
        np.array,
        lambda r_min, r_max, count: [random.randrange(r_min, r_max, 1) for i in range(count)],
        **gen_args
    )

    set_xy = np.column_stack((set_x, set_y))
    return set_xy


def tp1(**kwargs):

    plt.figure()


    required_args = [
        'r_min',
        'r_max',
        'count',
        'set_div_condition',
        'bchk_condition',
        'test_sample_size'
    ]
    for arg in required_args:
        if arg not in kwargs.keys():
            raise Exception("Missing arg : {}".format(arg))

    try:
        set_xy = gen_coords(
            r_min=kwargs['r_min'],
            r_max=kwargs['r_max'],
            count=kwargs['count']
        )

    except Exception as e:
        raise Exception("Error while generating random coordinates", e)

    set_xy = np.array([
        e for e in set_xy if kwargs['set_div_condition'](e)
    ])
    new_count = len(set_xy)

    train_benchmark = np.array(kwargs['bchk_condition'](set_xy))
    set_train, set_test, bchk_train, bchk_test = train_test_split(
        set_xy,
        train_benchmark,
        test_size=kwargs['test_sample_size']
    )

    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')

    plt.scatter(set_xy[:, 0], set_xy[:, 1])

    clf = RandomForestClassifier()
    clf.fit(set_train, bchk_train)

    y_pred = clf.predict(set_test)

    accuracy = accuracy_score(bchk_test, y_pred)
    print("Accuracy: {}".format(accuracy))

    to_predict = np.array([
        (1, 1),
        (2, 2),
        (300, 400),
        (-1, -2),
        (-30, -200)
    ])
    expected = [1, 1, 1, 0, 0]
    prediction = clf.predict(to_predict)
    accuracy = accuracy_score(prediction, expected)

    print(" --- Accuracy tests")
    print(" \\___ Expected answer", expected)
    print(" \\___ Model's answer", list(prediction))
    print(" \\___ Accuracy", accuracy)