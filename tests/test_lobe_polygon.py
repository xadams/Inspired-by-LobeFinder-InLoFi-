# coding=utf-8

"""
Tests for md_utils script
"""

import logging
import unittest
import os
import numpy as np
import shutil
import six
import errno
import csv
import sys
import difflib

from lobe_polygon import main, warning, TOL
from contextlib import contextmanager


# noinspection PyTypeChecker
def diff_lines(floc1, floc2, delimiter=","):
    """
    Determine all lines in a file are equal.
    This function became complicated because of edge cases:
        Do not want to flag files as different if the only difference is due to machine precision diffs of floats
    Thus, if the files are not immediately found to be the same:
        If not, test if the line is a csv that has floats and the difference is due to machine precision.
        Be careful if one value is a np.nan, but not the other (the diff evaluates to zero)
        If not, return all lines with differences.
    @param floc1: file location 1
    @param floc2: file location 1
    @param delimiter: defaults to CSV
    @return: a list of the lines with differences
    """
    diff_lines_list = []
    # Save diffs to strings to be converted to use csv parser
    output_plus = ""
    output_neg = ""
    with open(floc1, 'r') as file1:
        with open(floc2, 'r') as file2:
            diff = list(difflib.ndiff(file1.read().splitlines(), file2.read().splitlines()))

    for line in diff:
        if line.startswith('-') or line.startswith('+'):
            diff_lines_list.append(line)
            if line.startswith('-'):
                output_neg += line[2:] + '\n'
            elif line.startswith('+'):
                output_plus += line[2:] + '\n'

    if len(diff_lines_list) == 0:
        return diff_lines_list

    warning("Checking for differences between files {} {}".format(floc1, floc2))
    try:
        # take care of parentheses
        for char in ('(', ')', '[', ']'):
            output_plus = output_plus.replace(char, delimiter)
            output_neg = output_neg.replace(char, delimiter)
        # pycharm doesn't know six very well
        # noinspection PyCallingNonCallable
        diff_plus_lines = list(csv.reader(six.StringIO(output_plus), delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC))
        # noinspection PyCallingNonCallable
        diff_neg_lines = list(csv.reader(six.StringIO(output_neg), delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC))
    except ValueError:
        diff_plus_lines = output_plus.split('\n')
        diff_neg_lines = output_neg.split('\n')
        for diff_list in [diff_plus_lines, diff_neg_lines]:
            for line_id in range(len(diff_list)):
                diff_list[line_id] = [x.strip() for x in diff_list[line_id].split(delimiter)]

    if len(diff_plus_lines) == len(diff_neg_lines):
        # if the same number of lines, there is a chance that the difference is only due to difference in
        # floating point precision. Check each value of the line, split on whitespace or comma
        diff_lines_list = []
        for line_plus, line_neg in zip(diff_plus_lines, diff_neg_lines):
            if len(line_plus) == len(line_neg):
                # print("Checking for differences between: ", line_neg, line_plus)
                for item_plus, item_neg in zip(line_plus, line_neg):
                    try:
                        item_plus = float(item_plus)
                        item_neg = float(item_neg)
                        # if difference greater than the tolerance, the difference is not just precision
                        # Note: if only one value is nan, the float diff is zero!
                        #  Thus, check for diffs only if neither are nan; show different if only one is nan
                        diff_vals = False
                        if np.isnan(item_neg) != np.isnan(item_plus):
                            diff_vals = True
                            warning("Comparing '{}' to '{}'.".format(item_plus, item_neg))
                        elif not (np.isnan(item_neg) and np.isnan(item_plus)):
                            # noinspection PyTypeChecker
                            if not np.isclose(item_neg, item_plus, TOL):
                                diff_vals = True
                                warning("Values {} and {} differ.".format(item_plus, item_neg))
                        if diff_vals:
                            diff_lines_list.append("- " + " ".join(map(str, line_neg)))
                            diff_lines_list.append("+ " + " ".join(map(str, line_plus)))
                            break
                    except ValueError:
                        # not floats, so the difference is not just precision
                        if item_plus != item_neg:
                            diff_lines_list.append("- " + " ".join(map(str, line_neg)))
                            diff_lines_list.append("+ " + " ".join(map(str, line_plus)))
                            break
            # Not the same number of items in the lines
            else:
                diff_lines_list.append("- " + " ".join(map(str, line_neg)))
                diff_lines_list.append("+ " + " ".join(map(str, line_plus)))
    return diff_lines_list


@contextmanager
def capture_stdout(command, *args, **kwargs):
    # pycharm doesn't know six very well, so ignore the false warning
    # noinspection PyCallingNonCallable
    out, sys.stdout = sys.stdout, six.StringIO()
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
    sys.stdout = out


@contextmanager
def capture_stderr(command, *args, **kwargs):
    # pycharm doesn't know six very well, so ignore the false warning
    # noinspection PyCallingNonCallable
    err, sys.stderr = sys.stderr, six.StringIO()
    command(*args, **kwargs)
    sys.stderr.seek(0)
    yield sys.stderr.read()
    sys.stderr = err


def silent_remove(filename, disable=False):
    """
    Removes the target file name, catching and ignoring errors that indicate that the
    file does not exist.

    @param filename: The file to remove.
    @param disable: boolean to flag if want to disable removal
    """
    if not disable:
        try:
            os.remove(filename)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise


__author__ = 'xadams'

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# File Locations #

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
NO_NEW_OUTFILE = os.path.join(DATA_DIR, "single_outfile.csv")
NO_NEW_LISTFILE = os.path.join(DATA_DIR, "single_listfile.txt")
NEW_OUTFILE = os.path.join(DATA_DIR, "outfile.csv")
NEW_IMAGE_FILE = os.path.join(DATA_DIR, "col0_cut_01_Roi1.png")
IMAGE_FILE2 = os.path.join(DATA_DIR, "col0_cut_01_Roi2.png")
APPEND_LISTFILE = os.path.join(DATA_DIR, "double_listfile.txt")
APPEND_OUTFILE = os.path.join(DATA_DIR, "append_outfile.csv")

GOOD_SINGLE_OUTFILE = os.path.join(DATA_DIR, "single_outfile.csv")
GOOD_APPEND_OUTFILE = os.path.join(DATA_DIR, "good_append_outfile.csv")


# Tests

class TestMainFailWell(unittest.TestCase):
    def testHelp(self):
        test_input = ['-h']
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertFalse(output)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)

    def testNoSuchFile(self):
        test_input = ["-l", "ghost"]
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("Could not find list" in output)

    def testNoSpecifiedFile(self):
        test_input = []
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("No list file given" in output)

    def testNoNewFiles(self):
        test_input = ["-l", NO_NEW_LISTFILE]
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("already in the output" in output)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("listfile unreadable or already" in output)


class TestMain(unittest.TestCase):
    def testNewOutputFile(self):
        test_input = ["-l", NO_NEW_LISTFILE, "-o", NEW_OUTFILE]
        try:
            silent_remove(NEW_OUTFILE)
            main(test_input)
            self.assertFalse(diff_lines(NEW_OUTFILE, GOOD_SINGLE_OUTFILE))
        finally:
            silent_remove(NEW_OUTFILE, disable=DISABLE_REMOVE)
            silent_remove(NEW_IMAGE_FILE, disable=DISABLE_REMOVE)

    def testAppendOutputFile(self):
        test_input = ["-l", APPEND_LISTFILE, "-o", APPEND_OUTFILE]
        try:
            shutil.copyfile(GOOD_SINGLE_OUTFILE, APPEND_OUTFILE)
            main(test_input)
            self.assertFalse(diff_lines(APPEND_OUTFILE, GOOD_APPEND_OUTFILE))
        finally:
            silent_remove(APPEND_OUTFILE, disable=DISABLE_REMOVE)
            silent_remove(IMAGE_FILE2, disable=DISABLE_REMOVE)
