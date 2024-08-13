#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################
# Classical model
##################################################

import os
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import matplotlib.gridspec as gridspec

def find_equilibrium_in_range(func_1, func_2, xmin, xmax, steps=100):
    """
    Find the equilibrium point in a given range by finding the intersection of two functions.

    Parameters:
    - func_1 (function): The first function.
    - func_2 (function): The second function.
    - xmin (float): The minimum x-value of the range.
    - xmax (float): The maximum x-value of the range.

    Returns:
    - float or None: The x-coordinate of the equilibrium point if found, None otherwise.
    """
    x = np.linspace(xmin, xmax, steps)
    y_1 = func_1(x)
    y_2 = func_2(x)

    # Sometimes, the function we pass is a constant
    # Create an array of the same length as x
    if type(y_1) is int:
        y_1 = np.full_like(x, y_1)
    if type(y_2) is int:
        y_2 = np.full_like(x, y_2)

    # Find the intersection of the two functions
    intersection = np.argwhere(np.diff(np.sign(y_1 - y_2))).flatten()
    if len(intersection) == 0:
        return None
    else:
        return x[intersection[0]]

def solve_function_for_y(func, y, xmin, xmax,steps=100):
    """
    Solve a function for a given y-value.

    Parameters:
    - func (function): The function to solve.
    - y (float): The y-value to solve for.
    - steps (int, optional): The number of steps to use in the range of x.

    Returns:
    - float: The x-value that solves the function for the given y-value.
    """

    x = np.linspace(xmin, xmax, steps)
    y_vals = func(x)

    # Find the x-value that yields the desired y-value
    intersection = np.argwhere(np.diff(np.sign(y_vals - y))).flatten()
    if len(intersection) == 0:
        return None
    else:
        return x[intersection[0]]


def plot_multiple_functions(ax, xmin, xmax, funcs, annotate=None, add_lines=None, x_title='x', y_title='y', legend=False, steps=100, ymax=None, reverse_axes = False, outpath=None):
    """
    Plot multiple functions within a given range and optionally annotate points.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to plot on.
    - xmin (float): The minimum x-value for the plot.
    - xmax (float): The maximum x-value for the plot.
    - funcs (dict): A dictionary where keys are function objects and values are dictionaries 
                    with optional keys:
                        - 'color' (str, optional): The color of the function line.
                        - 'linestyle' (str, optional): The linestyle of the function line.
                        - 'linewdith' (float, optional): The linewidth of the function line.
                        - 'label' (str, optional): The label of the function line.

    - annotate (list of dict, optional): A list of dictionaries, each containing:
        - 'x' (float): The x-coordinate of the annotation.
        - 'x_max' (float): The maximum x-coordinate of the annotation line.
        - 'y_max' (float): The maximum y-coordinate of the annotation line.
        - 'x_label' (str): The label for the x-coordinate.
        - 'y_label' (str): The label for the y-coordinate.
        - 'func' (function): The function used to calculate the y-value.
        - 'color' (str, optional): The color of the annotation lines.
        - 'linestyle' (str, optional): The linestyle of the annotation lines.
        - 'linewidth' (float, optional): The linewidth of the annotation lines.

    - add_lines (list of dict, optional): A list of dictionaries, each containing:
        - 'coords' ((x_1, y_1), (x_2, y_2)): The start and end points of the line.
        - 'color' (str, optional): The color of the line.
        - 'linestyle' (str, optional): The linestyle of the line.
        - 'linewidth' (float, optional): The linewidth of the line.
    - steps (int, optional): The number of points to evaluate the functions at.
    - ymax (float, optional): The maximum y-value for the plot.
    - outpath (str, optional): The file path to save the plot. If None, the plot is shown but not saved.

    Raises:
    - ValueError: If xmin is not less than xmax or if annotate dictionaries lack required keys.
    - TypeError: If funcs is not a dictionary or if annotate is not a list.

    Returns:
    None
    """
    if not isinstance(xmin, (int, float)):
        raise TypeError("xmin must be an integer or float")
    if not isinstance(xmax, (int, float)):
        raise TypeError("xmax must be an integer or float")
    if xmin >= xmax:
        raise ValueError("xmin must be less than xmax")
    
    if not isinstance(funcs, dict):
        raise TypeError("funcs must be a dictionary")
    
    if annotate is not None and not isinstance(annotate, list):
        raise TypeError("annotate must be a list")

    x_vals = np.linspace(xmin, xmax, steps)

    for func, style in funcs.items():
        if not callable(func):
            raise TypeError("All keys in funcs must be callable functions")
        y_vals = func(x_vals)
        if type(y_vals) is int:
            y_vals = np.full_like(x_vals, y_vals)
        color = style.get('color', 'black')
        linestyle = style.get('linestyle', '-')
        linewidth = style.get('linewidth', 1)
        label = style.get('label', None)
        if reverse_axes == True:
            y_vals, x_vals = x_vals, y_vals
            
        ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, label=label, linewidth=linewidth)

    if annotate:
        xticks = []
        yticks = []
        xlabels = []
        ylabels = []
        for annotation in annotate:
            if not isinstance(annotation, dict):
                raise TypeError("Each annotation must be a dictionary")
            required_keys = {'x', 'x_label', 'y_label', 'func'}
            if not required_keys.issubset(annotation.keys()):
                raise ValueError(f"Each annotation must contain the keys: {required_keys}")
            if not callable(annotation['func']):
                raise TypeError("The 'func' in each annotation must be a callable function")
            
            x = np.array([annotation['x']])
            y = annotation['func'](x)
            x_label = annotation['x_label']
            y_label = annotation['y_label']

            if 'y_max' in annotation:
                y_max = np.array([annotation['y_max']])
            else:
                y_max = annotation['func'](x)

            if 'x_max' in annotation:
                x_max = np.array([annotation['x_max']])
            else:
                x_max = x

            if reverse_axes == True:
                x, y = y, x
                x_max, y_max = y_max, x_max
                x_label, y_label = y_label, x_label
            xticks.append(x)
            yticks.append(y)
            xlabels.append(x_label)
            ylabels.append(y_label)
            color = annotation.get('color', 'black')
            linestyle = annotation.get('linestyle', '-')
            ax.vlines(x=x, ymin=0, ymax=y_max, color=color, linestyle=linestyle)
            ax.hlines(y=y, xmin=0, xmax=x_max, color=color, linestyle=linestyle)

        xticks = np.array(xticks).flatten()
        yticks = np.array(yticks).flatten()
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

    if add_lines is not None:
        for line in add_lines:
            if not isinstance(line, dict):
                raise TypeError("Each line must be a dictionary")
            required_keys = {'coords'}
            if not required_keys.issubset(line.keys()):
                raise ValueError(f"Each line must contain the keys: {required_keys}")
            if not isinstance(line['coords'], tuple) or len(line['coords']) != 2:
                raise TypeError("Each line must contain a tuple of two coordinates")
            line['x_1'], line['y_1'] = line['coords'][0]
            line['x_2'], line['y_2'] = line['coords'][1]
            ax.plot([line['x_1'], line['x_2']], [line['y_1'], line['y_2']], color='black', linestyle='--')

    if reverse_axes == True:
        x_title, y_title = y_title, x_title
    if x_title is not None:
        ax.set_xlabel(x_title)
    if y_title is not None:
        ax.set_ylabel(y_title)

    if legend:
        ax.legend()

    if reverse_axes == True:
        xmin, xmax, ymin, ymax = 0, ymax, xmin, xmax
    ax.set(xlim=(xmin, xmax), ylim=(0, ymax))

    if outpath:
        plt.savefig(outpath, bbox_inches='tight')

def step_function(x_values, center_at, value_at_center):
    """
    Returns a step function based on the given x_values, center_at, and value_at_center.

    Parameters:
        x_values (list or numpy array): The x values for the step function.
        center_at (int or float): The center point of the step function.
        value_at_center (int or float): The value at the center point of the step function.

    Returns:
        numpy array: The step function values corresponding to the given x_values.

    Raises:
        TypeError: If x_values is not a list or numpy array, or if center_at or value_at_center is not an int or float.
        ValueError: If x_values is an empty list.
    """

    # Check if x_values is a list or numpy array
    if not isinstance(x_values, (list, np.ndarray)):
        raise TypeError("x_values must be a list or numpy array")

    # Check if center_at is an int or float
    if not isinstance(center_at, (int, float)):
        raise TypeError("center_at must be an int or float")

    # Check if value_at_center is an int or float
    if not isinstance(value_at_center, (int, float)):
        raise TypeError("value_at_center must be an int or float")

    # Check if x_values is not empty
    if len(x_values) == 0:
        raise ValueError("x_values cannot be an empty list")

    # Calculate the absolute difference between each value in x_values and the center
    difference_from_center = np.abs(np.array(x_values) - center_at)

    # Replace all values in x_values with -1 except the one closest to the center
    closest_to_center = np.where(difference_from_center == difference_from_center.min(), value_at_center, -1)

    return closest_to_center

################################
# Beginning of the script
################################

base_path = '/home/eric/Documents/Websites/eric-roca.github.io/'

# Colors
colors = {
    'supply' : {'normal' : 'darkorange', 'shifted' : 'moccasin'},
    'demand' : {'normal' : 'rebeccapurple', 'shifted' : 'plum'},
    'other' : 'lightsteelblue',
    'other_alt' : 'royalblue'
}

# IS curve
# Y = 10 + 0.6 (Y-3) + 1/r + 8
# r = 5/(2y-81)
is_curve = lambda x: 5/(2*x - 81)

outpath = os.path.join(base_path, 'static/img/macro_2/is-curve.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 44, 55, {
    # Functions
    is_curve : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS curve'},
    },
    x_title=r'Y Production',
    y_title=r"r Taux d'intérêt",
    legend = True,
    outpath = outpath
)
fig.savefig(outpath, bbox_inches='tight')

# Change in G, IS
# from
# Y = 10 + 0.6 (Y-3) + 1/r + 8
# r = 2.5/(2y-81)
# to
# Y = 10 + 0.6 (Y-3) + 1/r + 9
# r = 2.5/(y-43)

is_curve_1 = lambda x: 5/(2*x - 81)
is_curve_2 = lambda x: 2.5/(x-43)

outpath = os.path.join(base_path, 'static/img/macro_2/is-curve-change_g.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 44, 55, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS',
    },
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS, changement de G'},
    },
    annotate=[
        {'x' : 46, 'x_label' : '{:.2f}'.format(46), 'y_label' : '{:.2f}'.format(is_curve_1(46)), 'func' : is_curve_1, 'color' : colors['other'],'linestyle' : '--',},
        {'x': 48.5, 'x_label' : '{:.2f}'.format(48.5), 'y_label' : '{:.2f}'.format(is_curve_2(48.5)), 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 50, 'x_label' : '{:.2f}'.format(50), 'y_label' : '{:.2f}'.format(is_curve_1(50)), 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '-.'},
        {'x' : 52.5, 'x_label' : '{:.2f}'.format(52.5), 'y_label' : '{:.2f}'.format(is_curve_2(52.5)), 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '-.'},],
    x_title=r'Y Production',
    y_title=r"r Taux d'intérêt",
    legend = True,
    outpath = outpath
)
fig.savefig(outpath, bbox_inches='tight')


# Change in T, IS
# Y = 10 + 0.6 (Y-3) + 1/r + 8
# r = 5/(2y-81)
# to
# Y = 10 + 0.6 (Y-4) + 1/r + 8
# r = 2.5/(y-39)

is_curve_1 = lambda x: 5/(2*x - 81)
is_curve_2 = lambda x: 2.5/(x-39)

outpath = os.path.join(base_path, 'static/img/macro_2/is-curve-change_g.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 44, 55, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS',
    },
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS, changement de T'},
    },
    annotate=[
        {'x' : 47.5, 'x_label' : '{:.2f}'.format(47.5), 'y_label' : '{:.2f}'.format(is_curve_1(47.5)), 'func' : is_curve_1, 'color' : colors['other'],'linestyle' : '--',},
        {'x': 46, 'x_label' : '{:.2f}'.format(46), 'y_label' : '{:.2f}'.format(is_curve_2(46)), 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 51.5, 'x_label' : '{:.2f}'.format(51.5), 'y_label' : '{:.2f}'.format(is_curve_1(51.5)), 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '-.'},
        {'x' : 50, 'x_label' : '{:.2f}'.format(50), 'y_label' : '{:.2f}'.format(is_curve_2(50)), 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '-.'},],
    x_title=r'Y Production',
    y_title=r"r Taux d'intérêt",
    legend = True,
    outpath = outpath
)
fig.savefig(outpath, bbox_inches='tight')

# LM curve
# M = 2Y/r^0.5 --> r = (2Y/M)^2
lm_curve = lambda x: (2*x/4)**2

outpath = os.path.join(base_path, 'static/img/macro_2/lm-curve.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 5, {
    # Functions
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM curve'},
    },
    x_title=r'$Y$ Production',
    y_title=r"$r$ Taux d'intérêt",
    legend = True,
    outpath = outpath
)
fig.savefig(outpath, bbox_inches='tight')

# Change in LM
# Increase in Ms

# from 
# M = 2Y/r^0.5 --> r = (2Y/M)^2, M = 4
# to
# M = 2Y/r^0.5 --> r = (2Y/M)^2, M = 5

lm_curve_1 = lambda x: (2*x/4)**2
lm_curve_2 = lambda x: (2*x/5)**2

# In my example, the multiplier is
# 1/L_y = r^0.5/2
# with r initially being (2Y/4)^2 we get
# 1/L_y = Y/4
# evaluated at Y = 2.5 it gives 0.625
# Hence, we move the X point from 2.5 to 3.125
# We can get it more easily by first computing the interest rate at y=2.5
# r = (2 * 2.5 / 4)^2 = 1.5625
# The multiplier is then
# r^0.5/2 = 1.5625^2/2 = 0.625 

outpath = os.path.join(base_path, 'static/img/macro_2/lm-curve-change_m.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 5, {
    # Functions
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM, changement de M'},
    },
    annotate=[
        {'x' : 2.5, 'x_label' : '{:.2f}'.format(2.5), 'y_label' : '{:.2f}'.format(lm_curve_1(2.5)), 'func' : lm_curve_1, 'color' : colors['other'],'linestyle' : '--',},
        {'x': 3.125, 'x_label' : '{:.2f}'.format(3.125), 'y_label' : '{:.2f}'.format(lm_curve_2(3.125)), 'func' : lm_curve_2, 'color' : colors['other'], 'linestyle' : '--'},],
    x_title=r'$Y$ Production',
    y_title=r"$r$ Taux d'intérêt",
    legend = True,
    outpath = outpath
)        

# IS-LM together
is_curve = lambda x: 5/(2*x**0.5)
lm_curve = lambda x: (2*x/4)**3

y_eq = find_equilibrium_in_range(is_curve, lm_curve, 0, 10, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/is-lm.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 5, {
    # Functions
    is_curve : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS'},
    lm_curve : { 
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM'},
    },
    annotate=[
        {'x' : y_eq, 'x_label' : r'$Y^\star$', 'y_label' : r'$r^\star$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},],
    x_title=r'$Y$ Production',
    y_title=r"$r$ Taux d'intérêt",
    legend = True,
    outpath = outpath
)

# Ex 2
# IS: Y = 3 + 0.7(Y-2)+2-0.4r+4
# r = (7.6 - 0.3Y)/0.4 = 19-0.75Y
# LM: 30 = 3Y- 2r
# r = (3Y-30)/2

is_curve = lambda x: 19 - 0.75*x
lm_curve = lambda x: (3*x-30)/2

y_eq = find_equilibrium_in_range(is_curve, lm_curve, 10, 20, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/is-lm-2.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 10, 20, {
    # Functions
    is_curve : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS'},
    lm_curve : { 
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM'},
    },
    annotate=[
        {'x' : y_eq, 'x_label' : r'$Y^\star$', 'y_label' : r'$r^\star$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},],
    x_title=r'$Y$ Production',
    y_title=r"$r$ Taux d'intérêt",
    legend = True,
    outpath = outpath
)

# Ex 3
# IS with fixed taxes
# Y = 10 + 0.8(Y-5) + 1/r + 4
# r = 5/(Y-50)

# IS with variable taxes
# Y = 10 + 0.8(Y - 0.2 Y - 5) + 1/r + 4
# r = 25/(9Y-250)
is_curve_1 = lambda x: 5/(x-50)
is_curve_2 = lambda x: 5/(x-50+4 * x * 0.002)

outpath = os.path.join(base_path, 'static/img/macro_2/is-curve-change_t.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 51, 55, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS, T proportionnel'},
    },
    x_title=r'Y Production',
    y_title=r"r Taux d'intérêt",
    legend = True,
    outpath = outpath
)

# Ex 6
# LM is 50 = 2*(Y-T)+0.2/r
# r = 0.1/(25-y+t)

lm_tax_0 = lambda x: 0.1/(25-x)
lm_tax_1 = lambda x: 0.1/(25-x+1)

outpath = os.path.join(base_path, 'static/img/macro_2/lm-curve-change_t.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 20, {
    # Functions
    lm_tax_0 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM t=0'},
    lm_tax_1 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM t=1'},
    },
    x_title=r'Y Production',
    y_title=r"$r$ Taux d'intérêt",
    legend = True,
    outpath = outpath
)