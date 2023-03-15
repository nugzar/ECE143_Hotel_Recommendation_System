from typing import Union

import numpy as np


def __convert_int__(str_in: Union[str, int]) -> Union[str, int]:
    """
        Convert a string to an integer, if possible.

        If the input is already an integer, return it unchanged.

        If the input is a string, strip any extraneous whitespace from the input
        string, remove commas, and replace the substring '\\N' with -1.

        Then, attempt to convert the cleaned string to an integer.

        If the conversion is successful, return the integer value. Otherwise, return NAN
    """
    assert isinstance(str_in,(int, str)), "Inputted value is not a string or integer"
    if isinstance(str_in, int):
        return str_in
    str_in = str_in.strip().replace(',', '').replace('\\N', '-1')
    return int(str_in) if str_in.isdigit() else np.nan