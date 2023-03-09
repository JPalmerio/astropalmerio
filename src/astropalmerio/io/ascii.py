# -*- coding: utf-8 -*-

__all__ = [
    "read_nddata",
]

import logging
import numpy as np

log = logging.getLogger(__name__)


def read_nddata(
    filename,
    column_nb,
    end=None,
    err=True,
    dtype=float,
    single_err=False,
    sep=None,
    strip=None,
    verbose=False,
):
    """
    Helper function that reads a text file with the correct format and
    returns the data, with errors and upper or lower limits in the form:
        data[0] = data        (float by default)
        data[1] = plus error  (float by default)
        data[2] = minus error (float by default)
        data[3] = upper limit (float by default)
        data[4] = lower limit (float by default)
    If there is no data it fills the errors with NaN.
    Otherwise if the data exists but it doesn't find errors it will set them to 0.
    Upper and lower limits and converted to 1 for True, and 0 for False in the requested dtype.

    Parameters
    ----------
    filename : str, pathlib.Path
        Filename to read.
    column_nb : int
        Column number, 0-indexed.
    end : int, optional
        Maximum number of lines to read.
    err : bool, optional
        If err is True it assumes format of file is:
        1) data, 2) error plus, 3) error minus
        If err is False, it sets them to 0.
    dtype : data-type, optional
        Data type in which to cast the resulting array.
    single_err : bool, optional
        If single_err is True it assumes format of file is:
        1) data, 2) error
    sep : str, optional
        The string used to separate values (passed to str.split()).
    strip : str, optional
        The string used to strip the lines (passed to str.strip()).
    verbose : bool, optional
        Activate verbosity.

    Returns
    -------
    out : ndarray
        Data read from the text file in the following format:
            data[0] = data
            data[1] = plus error
            data[2] = minus error
            data[3] = upper limit
            data[4] = lower limit
    """
    data = [[], [], [], [], []]
    with open(filename, "r", encoding="utf-8") as f:
        i = 1
        i_counter = 0
        if err is True:
            for line in f:
                if line[0] != "#" and line[0] != "!" and line[0] != "%":
                    i_counter += 1
                    line = line.strip(strip)
                    columns = line.split(sep)
                    try:
                        # Check for upper limit
                        if "<" in columns[column_nb]:
                            # Remove '<' from data
                            place = columns[column_nb].find("<")
                            columns[column_nb] = columns[column_nb][place + 1 :]
                            # Append upper limit
                            data[3].append(True)
                            # Set errors to 0.0, and lower limit to false
                            data[1].append(0.0)  # error plus
                            data[2].append(0.0)  # error minus
                            data[4].append(False)

                        # Check for lower
                        elif ">" in columns[column_nb]:
                            # Remove '>' from data
                            place = columns[column_nb].find(">")
                            columns[column_nb] = columns[column_nb][place + 1 :]
                            # Append lower limit
                            data[4].append(True)
                            # Set errors to 0.0, and upper limit to false
                            data[1].append(0.0)  # error plus
                            data[2].append(0.0)  # error minus
                            data[3].append(False)

                        # if no lower or upper limits, append errors
                        else:
                            data[3].append(False)
                            data[4].append(False)
                            try:
                                data[1].append(
                                    float(columns[column_nb + 1])
                                )  # error plus
                            except IndexError:
                                data[1].append(None)  # error plus
                            except ValueError:
                                data[1].append(0.0)  # error plus
                            try:
                                data[2].append(
                                    np.abs(float(columns[column_nb + 2]))
                                )  # error minus
                            except IndexError:
                                data[2].append(None)  # error minus
                            except ValueError:
                                data[2].append(0.0)  # error minus

                        # Append data
                        try:
                            data[0].append(dtype(columns[column_nb]))
                        except ValueError:
                            if verbose:
                                log.warning(
                                    "In read_data : Couldn't convert %s to %s for column %d, line %d in file %s."
                                    " Replacing with NaN.",
                                    columns[column_nb],
                                    dtype,
                                    column_nb,
                                    i,
                                    filename,
                                )
                            data[0].append(np.nan)
                    # If no data
                    except IndexError:
                        if verbose:
                            log.warning(
                                "In read_data : No data found for column %d, line %d in file %s."
                                " Input will be NaN.",
                                column_nb,
                                i,
                                filename,
                            )
                        data[0].append(np.nan)  # data
                        data[1].append(np.nan)  # error plus
                        data[2].append(np.nan)  # error minus
                        data[3].append(np.nan)  # upper limit
                        data[4].append(np.nan)  # lower limit
                i += 1
                if (end is not None) and (i_counter >= end):
                    break
        else:
            for line in f:
                if line[0] != "#" and line[0] != "!" and line[0] != "%":
                    i_counter += 1
                    line = line.strip(strip)
                    columns = line.split(sep)
                    try:
                        # Check for upper limit
                        if "<" in columns[column_nb]:
                            # Remove '<' from data
                            place = columns[column_nb].find("<")
                            columns[column_nb] = columns[column_nb][place + 1 :]
                            # Append upper limit
                            data[3].append(True)
                            # Set lower limit to false
                            data[4].append(False)

                        # Check for lower
                        elif ">" in columns[column_nb]:
                            # Remove '>' from data
                            place = columns[column_nb].find(">")
                            columns[column_nb] = columns[column_nb][place + 1 :]
                            # Append lower limit
                            data[4].append(True)
                            # Set upper limit to false
                            data[3].append(False)

                        else:
                            data[3].append(False)
                            data[4].append(False)

                        # Append data
                        try:
                            data[0].append(dtype(columns[column_nb]))
                        except ValueError:
                            if verbose:
                                log.warning(
                                    "In read_data : Couldn't convert %s to %s for column %d, line %d in file %s."
                                    " Replacing with NaN.",
                                    columns[column_nb],
                                    dtype,
                                    column_nb,
                                    i,
                                    filename,
                                )
                            data[0].append(np.nan)
                        data[1].append(0.0)  # error plus
                        data[2].append(0.0)  # error minus

                    except IndexError:
                        data[0].append(np.nan)  # data
                        data[1].append(np.nan)  # error plus
                        data[2].append(np.nan)  # error minus
                        data[3].append(np.nan)  # upper limit
                        data[4].append(np.nan)  # lower limit
                if (end is not None) and (i_counter >= end):
                    break
    if single_err:
        data[2] = data[1]

    data[0] = np.array(data[0]).astype(dtype)
    data[1] = np.array(data[1]).astype(dtype)
    data[2] = np.array(data[2]).astype(dtype)
    data[3] = np.asarray(data[3]).astype(dtype)
    data[4] = np.asarray(data[4]).astype(dtype)
    data = np.asarray(data)
    return data
