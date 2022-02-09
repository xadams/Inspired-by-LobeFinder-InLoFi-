# This script reads in coordinate files for cells and does the following:
# 1. Constructs the lobes
# 2. Eliminates lobes below a threshold
# 3. Outputs desired metrics including:
# a. Number of lobes (indirectly)
# b. Width of lobes
# c. Height of lobes
# Current Problems:
# i. I haven't interpolated, so the measurements are somewhat sensitive to the coordinate file
# ii. Oddly shaped lobes can be split in two
# iii. ~10% of lobes aren't draw quite correctly
from shapely.ops import polylabel
import shapely.geometry as geo
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import math
import pandas as pd
from descartes.patch import PolygonPatch
from scipy.signal import find_peaks

# Error Codes
# The good status code
GOOD_RET = 0
INPUT_ERROR = 1
IO_ERROR = 2
INVALID_DATA = 3
TOL = 0.0000000001
COLUMN_NAMES = ["filename", "lobe height", "lobe width", "neck width", "cell area", "lobe area", "area ratio"]


class MdError(Exception):
    pass


class InvalidDataError(MdError):
    pass


def file_rows_to_list(c_file):
    """
    Given the name of a file, returns a list of its rows, after filtering out empty rows
    @param c_file: file location
    @return: list of non-empty rows
    """
    with open(c_file) as f:
        row_list = [row.strip() for row in f.readlines()]
        return list(filter(None, row_list))


def warning(*objs):
    """Writes a message to stderr."""
    print("WARNING: ", *objs, file=sys.stderr)


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    'argv' is a list of arguments, or 'None' for 'sys.argv[1:]".
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(
        description='Process ROI coordinate files and output quantification and images.')
    parser.add_argument("-l", "--list", help="File with list of trajectory files")
    parser.add_argument("-o", "--out_file", help="File to write output to. Default is lobe_quantities.csv.",
                        type=str, default='lobe_quantities.csv')

    args = None
    # Try to parse command line arguments
    try:
        args = parser.parse_args(argv)
        args.files = []
        # If output file already exists, check existing files against input list
        if os.path.isfile(args.out_file):
            out_frame = pd.read_csv(args.out_file)
            logged_files = list(out_frame["filename"].unique())
        # Otherwise, generate output file with specified name
        else:
            print("Did not find {}, writing new file.".format(args.out_file))
            logged_files = []
            df = pd.DataFrame(columns=COLUMN_NAMES)
            df.to_csv(args.out_file)
        # For each file in listfile, check that it exists and has not already been processed
        if args.list:
            if os.path.isfile(args.list):
                args.files += file_rows_to_list(args.list)
                for file in args.files:
                    if not os.path.isfile(file):
                        print("Problems reading file: {}. Removing from list.".format(file))
                        args.files.remove(file)
                    elif file in logged_files:
                        args.files.remove(file)
                        print("Removing file: {}, which is already in the output file.".format(file))
                    if not args.files:
                        raise IOError("All files in listfile unreadable or already in csv file.")
            else:
                raise IOError("Could not find list file: {}".format(args.list))
        else:
            raise InvalidDataError("No list file given. "
                                   "Specify list file as described in the help documentation ('-h').")
    except IOError as e:
        warning("Problems reading file:", e)
        parser.print_help()
        return args, IO_ERROR
    except (KeyError, InvalidDataError, SystemExit) as e:
        if hasattr(e, 'code') and e.code == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR
    return args, GOOD_RET


def main(argv=None):
    # Options
    make_lobes = True  # Generate polygons for individual lobes and remove small lobes
    new_algorithm = True
    quantify_lobes = True  # Determine and output lobe count, height, and width
    plot_heights = True
    lobe_perimeter_cutoff = 0.5
    end_connectivity_cutoff = 0.5  # Fraction of points that the 'end' needs to 'see'
    lobe_angle_change = 30  # Minimum angle between necks for a new lobe
    # Read input
    args, ret = parse_cmdline(argv)

    if ret != GOOD_RET or args is None:
        return ret
    cellfiles = args.files

    # Iterate through cell files, placing each on a different plot
    df = pd.read_csv(args.out_file)
    for cellfile in cellfiles:
        fig, ax = plt.subplots()

        with open(cellfile, 'r') as fin:
            intable = [row.strip().split(' ')[1:] for row in fin]
            intable.append(intable[0])

        # Generate a polygon object from cell coordinates
        cell = geo.Polygon(np.asarray(intable).astype(int))
        cell_area = cell.area
        lobe_area = 0
        hull = cell.convex_hull
        # hull_patch = PolygonPatch(hull, alpha=0.5)
        # ax.add_patch(hull_patch)
        hull_distances = []
        for point in cell.exterior.coords:
            hull_distances.append(hull.exterior.distance(geo.Point(point)))
        peak_indices, empty_dict = find_peaks(hull_distances, height=2)
        if hull_distances[0] > hull_distances[1] and hull_distances[0] > hull_distances[-2]:
            peak_indices = np.append([0], peak_indices)
        # Extract coordinates as iterable lists
        cell_x, cell_y = cell.exterior.xy
        # plt.scatter(np.asarray(cell_x)[peak_indices], np.asarray(cell_y)[peak_indices])

        # Plot the cell border
        plt.plot(cell_x, cell_y, color='black', label='cell wall')

        if new_algorithm:
            # New core algorithm for determining "core" of the cell
            interior_x = list(np.asarray(cell_x)[peak_indices])
            interior_y = list(np.asarray(cell_y)[peak_indices])

        else:
            # Old algorithm for determining the "core" of the cell
            interior_x = []
            interior_y = []
            i = 0
            j = 2
            prev_line_length = 0
            # This boolean tracks if the furthest point has been passed
            past_max = False
            number_outside = 0
            while i+j < len(cell_x):
                a = geo.LineString([(cell_x[i], cell_y[i]), (cell_x[i + j], cell_y[i + j])])
                # If a local maximum has been reached, check if this point is a local minimum
                if a.length > prev_line_length + 5 and past_max:
                    interior_x.append(cell_x[i + j - 1])
                    interior_y.append(cell_y[i + j - 1])
                    i = i + j - 1
                    j = 2
                    past_max = False
                    prev_line_length = 0
                    number_outside = 0
                # If the connecting line is completely within the cell, move to the next point
                elif a.within(cell):
                    j += 1
                    if a.length < prev_line_length:
                        past_max = True
                    prev_line_length = a.length
                # Forgive the first point that falls outside the cell, but remember the point to come back to
                elif number_outside == 1:
                    interior_x.append(cell_x[i + j - 2])
                    interior_y.append(cell_y[i + j - 2])
                    i = i + j - 2
                    j = 2
                    past_max = False
                    prev_line_length = 0
                    number_outside = 0
                # Otherwise, write the point because the next line falls outside the cell
                else:
                    number_outside += 1
                    j += 1
                    if a.length < prev_line_length:
                        past_max = True
                    prev_line_length = a.length

        # Duplicate the first point to plot a closed cell
        interior_x.append(interior_x[0])
        interior_y.append(interior_y[0])
        plt.title(cellfile)

        # This loop forms the individual lobes based on the necks from the previous step
        # interior_x and interior_y are lists of the x and y coordinates of the "core" cell (without lobes)
        # cell is the polygon of the cell, and x and y are the exterior coordinates of it
        if make_lobes:
            j = 0  # index of front edge of lobe
            k = 0  # index of back edge of lobe
            all_lobes_x = []
            all_lobes_y = []
            lobes_x = []
            lobes_y = []
            extra_x = []
            extra_y = []
            first_lobe = True  # boolean signalling to save points for later
            in_lobe = False  # boolean signalling that next non-core ends the lobe
            back_to_core = False  # boolean signally that we are in the far edge of lobe
            core = list(zip(interior_x, interior_y))
            # Loop is written as such to not lose progress iterating along the cell
            while j+k < len(cell_x):
                # If the point is on the "core" of the cell, it may be the start of a lobe
                if (cell_x[j+k], cell_y[j+k]) in core:
                    if in_lobe:
                        back_to_core = True
                        lobes_x.append(cell_x[j+k])
                        lobes_y.append(cell_y[j+k])
                        k += 1
                    else:
                        lobes_x.append(cell_x[j])
                        lobes_y.append(cell_y[j])
                        j += 1

                elif back_to_core:
                    k = 0
                    back_to_core = False
                    in_lobe = False
                    all_lobes_x.append(lobes_x)
                    all_lobes_y.append(lobes_y)
                    lobes_x = []
                    lobes_y = []
                elif lobes_x:
                    lobes_x.append(cell_x[j])
                    lobes_y.append(cell_y[j])
                    j += 1
                    in_lobe = True
                    first_lobe = False
                else:
                    j += 1
                # Because we may start partially through a lobe, save these points for later
                if first_lobe:
                    extra_x.append(cell_x[j])
                    extra_y.append(cell_y[j])

            # Add points from the beginning of the cell to the final lobe
            for ex_x, ex_y in zip(extra_x, extra_y):
                lobes_x.append(ex_x)
                lobes_y.append(ex_y)
            all_lobes_x.append(lobes_x)
            all_lobes_y.append(lobes_y)

            lobes = []
            combine_lobes_index = []
            legend = True
            #  Create a polygon object for each lobe and measure the area
            for lobe_x, lobe_y in zip(all_lobes_x, all_lobes_y):
                coords = [[x, y] for x, y in zip(lobe_x, lobe_y)]
                # Iterate through lobe looking for a point that can 'see' 50% of points
                for i, (x, y) in enumerate(coords):
                    point = geo.Point(x, y)
                    num_in_cell = 0
                    for compare_x, compare_y in coords[i+1:]:
                        a = geo.LineString([point, (compare_x, compare_y)])
                        if a.within(cell):
                            num_in_cell += 1
                    try:
                        if num_in_cell/len(coords[i+1:]) >= end_connectivity_cutoff:
                            coords = coords[i:]
                            break
                    except ZeroDivisionError:
                        break

                rev_coords = coords[::-1]
                for i, (x, y) in enumerate(rev_coords):
                    point = geo.Point(x, y)
                    num_in_cell = 0
                    for compare_x, compare_y in rev_coords[i+1:]:
                        a = geo.LineString([point, (compare_x, compare_y)])
                        if a.within(cell):
                            num_in_cell += 1
                    try:
                        if num_in_cell/len(coords[i+1:]) >= end_connectivity_cutoff:
                            coords = coords[:len(coords)-i]
                            break
                    except ZeroDivisionError:
                        break

                try:
                    p = geo.Polygon(coords)
                    # Cutoff value for small lobes
                    if p.length > lobe_perimeter_cutoff:
                        if p.exterior.within(cell):
                            lobes.append(p)
                            lobe_area += p.area
                            x, y = p.exterior.xy
                            # plt.plot(x, y) # Plot each lobe
                            if legend:
                                plt.plot(x[-2:], y[-2:], color='#ED553B', label='Neck width')  # Plot only neck
                            else:
                                plt.plot(x[-2:], y[-2:], color='#ED553B')
                            try:
                                neck_slope = (y[-3] - y[0]) / (x[-3] - x[0])
                            except ZeroDivisionError:
                                neck_slope = 10e6
                            if 'prev_neck_slope' in locals():
                                try:
                                    lobe_neck_angle = np.arctan((neck_slope - prev_neck_slope)
                                                                / (1 + neck_slope * prev_neck_slope))
                                except ZeroDivisionError:
                                    lobe_neck_angle = 0
                                combine_lobes_index.append(lobe_neck_angle * 180 / np.pi)
                                # if lobe_neck_angle < lobe_angle_change:
                                #     combine_lobes_index.append(1)
                                # else:
                                #     combine_lobes_index.append(2)
                            prev_neck_slope = neck_slope
                            legend = False
                except ValueError:
                    print("Eliminating lobe with too few members.")

            # print(combine_lobes_index)
            legend = True
            ratio = lobe_area / cell_area
            if quantify_lobes:
                # Calculate and plot the height of each lobe
                for j, lobe in enumerate(lobes):
                    lobe_x, lobe_y = lobe.exterior.xy
                    d_max = 0
                    # Calculate the distance of each point from the neck to determine lobe height
                    for i, (x, y) in enumerate(zip(lobe_x[1:-2], lobe_y[1:-2]), 1):
                        x1 = lobe_x[0]
                        x2 = lobe_x[-2]
                        y1 = lobe_y[0]
                        y2 = lobe_y[-2]
                        p1 = np.asarray([x1, y1])
                        p2 = np.asarray([x2, y2])
                        p3 = np.asarray([x, y])
                        d = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
                        neck_length = math.hypot(x2 - x1, y2 - y1)
                        if d > d_max:
                            d_max = d
                            ind = i
                    # Replace vertical slopes with sufficiently large value for plotting
                    if plot_heights:
                        try:
                            neck_slope = (y2 - y1) / (x2 - x1)
                        except ZeroDivisionError:
                            neck_slope = 10e6
                        # Prevent divide by zero if the neck is horizontal
                        try:
                            neck_antislope = -1 / neck_slope
                        except ZeroDivisionError:
                            neck_antislope = 10e6
                        x_intercept = (y2 - lobe_y[ind] + neck_antislope * lobe_x[ind] - neck_slope * x2)\
                            / (neck_antislope - neck_slope)
                        y_intercept = neck_antislope * (x_intercept - lobe_x[ind]) + lobe_y[ind]
                        if legend:
                            plt.plot([lobe_x[ind], x_intercept], [lobe_y[ind], y_intercept],color='#3CAEA3', label='Lobe height')
                        else:
                            plt.plot([lobe_x[ind], x_intercept], [lobe_y[ind], y_intercept], color='#3CAEA3')

                    # Calculate the diameter of the inscribed circle
                    try:
                        inscribed_circle = polylabel(lobe, tolerance=0.1)
                        d = lobe.exterior.distance(inscribed_circle)
                        if legend:
                            circle = plt.Circle([inscribed_circle.x, inscribed_circle.y], d,
                                            facecolor='white', edgecolor='#20639B', lw=2, label='Lobe width')
                        else:
                            circle = plt.Circle([inscribed_circle.x, inscribed_circle.y], d,
                                                facecolor='white', edgecolor='#20639B', lw=2)
                        ax.add_patch(circle)
                        plt.text(inscribed_circle.x, inscribed_circle.y, j)
                    # TODO: change to specfic error type
                    except:
                        print("Polylabel error. Take a closer look at {}".format(cellfile))
                        d = 0

                    lobe_frame = pd.DataFrame([[cellfile, d_max, d, neck_length, cell_area, lobe_area, ratio]], columns=COLUMN_NAMES)
                    cell_area = ""
                    lobe_area = ""
                    ratio = ""
                    df = df.append(lobe_frame, ignore_index=True, sort=False)
                    legend = False

        image_name = "{}.png".format(os.path.splitext(cellfile)[0])
        plt.legend(loc='best')
        plt.savefig(image_name)
        fig.clear()
        plt.close()
    df.drop(df.columns[0], axis=1, inplace=True)
    df.to_csv(args.out_file)


if __name__ == '__main__':
    status = main()
    sys.exit(status)
