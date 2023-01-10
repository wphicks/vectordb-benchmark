from typing import List
import math


class ParametersBase:
    @staticmethod
    def least_common_multiple(args: List[int]):
        def lcm(a: int, b: int):
            return int(a * b / math.gcd(a, b))

        if len(args) == 0:
            return 0
        elif len(args) == 1:
            return args[0]
        else:
            _lcm = args[0]
            for i in range(1, len(args)):
                _lcm = lcm(_lcm, args[i])
            return _lcm
