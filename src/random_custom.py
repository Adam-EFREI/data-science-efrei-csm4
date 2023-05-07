# Gestion de l'aleatoire
import numpy as np
from sys import stderr
from typing import Any, Callable, List


class RandomCustom:

    def __init__(self):
        pass

    def gen_rand(self, ret_type: type, gen_routine: Callable, **params) -> Any:

        # Step 1 : Call the 'gen_routine' method
        try:
            result = gen_routine(**params)

        except Exception as e:
            print("Could not execute 'gen_rand' : {}".format(e), file=stderr)
            raise Exception

        # Step 2 : Make sure the 'gen_routine' result is of type [float]
        if type(result) is not list: # @Todo : Fix
            raise Exception("Could not execute 'gen_rand' : gen_routine should return [float]")

        # Step 3 : Transform the result into type 'ret_type'
        match ret_type:
            case np.array:
                result = np.array(result)

            case _:
                pass

        return result
