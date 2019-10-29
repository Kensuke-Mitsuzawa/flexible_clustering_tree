#! -*- coding: utf-8 -*-
from typing import List, Tuple
from collections import Counter
from itertools import chain


def default_fun_string_aggregation(input_lists: List[str]) -> List[Tuple[str, int]]:
    c_obj = Counter(input_lists)
    return c_obj.most_common(3)
