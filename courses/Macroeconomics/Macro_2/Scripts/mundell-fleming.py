#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################
# IS LM Policy
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


def plot_multiple_functions(ax, xmin, xmax, funcs, annotate=None, add_lines=None, x_title='x', y_title='y', legend=False, steps=100, ymin=0, ymax=None, reverse_axes = False, outpath=None):
    """
    Plot multiple functions within a given range and optionally annotate points.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to plot on.
    - xmin (float): The minimum x-value for the plot.
    - xmax (float): The maximum x-value for the plot.
    - funcs (dict): A dictionary where keys are function objects and values are dictionaries 
                    with optional keys:
                        - 'color' (str, optional): The color of the function line.
                        - 'alpha' (float, optional): The transparency of the function line.
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
        - 'alpha' (float, optional): The transparency of the annotation lines.
        - 'linestyle' (str, optional): The linestyle of the annotation lines.
        - 'linewidth' (float, optional): The linewidth of the annotation lines.

    - add_lines (list of dict, optional): A list of dictionaries, each containing:
        - 'coords' ((x_1, y_1), (x_2, y_2)): The start and end points of the line.
        - 'color' (str, optional): The color of the line.
        - 'linestyle' (str, optional): The linestyle of the line.
        - 'linewidth' (float, optional): The linewidth of the line.
    - steps (int, optional): The number of points to evaluate the functions at.
    - ymin (float, optional): The minimum y-value for the plot. Default is 0.
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
        alpha = style.get('alpha', 1.0)
        linestyle = style.get('linestyle', '-')
        linewidth = style.get('linewidth', 1)
        label = style.get('label', None)
        if reverse_axes == True:
            y_vals, x_vals = x_vals, y_vals
            
        ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, alpha = alpha, label=label, linewidth=linewidth)

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
            alpha = annotation.get('alpha', 1.0)
            ax.vlines(x=x, ymin=0, ymax=y_max, color=color, linestyle=linestyle, alpha=alpha)
            ax.hlines(y=y, xmin=0, xmax=x_max, color=color, linestyle=linestyle, alpha=alpha)

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
        xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

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


# Mundell-Fleming IS curve
is_curve = lambda x: 5/(2*x - 81)

outpath = os.path.join(base_path, 'static/img/macro_2/is_mf.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 90, 200, {
    # Functions
    is_curve : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}$'},
    },
    annotate=[],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    outpath=outpath
)

lm_curve = partial(step_function, center_at=140, value_at_center=0.1)

outpath = os.path.join(base_path, 'static/img/macro_2/lm_mf.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 90, 200, {
    # Functions
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'},
    },
    annotate=[],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=0.1,
    outpath=outpath
)

eq = find_equilibrium_in_range(is_curve, lm_curve, 90, 200, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/eq_mf.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 90, 200, {
    # Functions
    is_curve : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}$'},
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'},
    },
    annotate=[
        {'x' : eq, 'x_label' : r'$Y^\star$', 'y_label' : r'$e^\star$', 'func' : is_curve, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=0.1,
    steps=1000,
    outpath=outpath
)

# Floating exchange rate
## Change in G
is_curve_1 = lambda x: 2/(x - 11)
is_curve_2 = lambda x: 2/(x - 63)

lm_curve = partial(step_function, center_at=140, value_at_center=0.1)

eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve, 90, 200, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve, 90, 200, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/mf_floating_g.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 90, 200, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_2$'},
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'},
    },
    annotate=[
        {'x' : eq_1, 'x_label' : r'$Y^\star_1 = Y^\star_2$', 'y_label' : r'$e^\star_1$', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y^\star_1 = Y^\star_2$', 'y_label' : r'$e^\star_2$', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=0.1,
    steps=1000,
    outpath=outpath
)

## Change in Ms
is_curve = lambda x: 2/(x - 63)

lm_curve_1 = partial(step_function, center_at=120, value_at_center=0.1)
lm_curve_2 = partial(step_function, center_at=160, value_at_center=0.1)

eq_1 = find_equilibrium_in_range(is_curve, lm_curve_1, 90, 200, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve, lm_curve_2, 90, 200, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/mf_floating_ms.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 90, 200, {
    # Functions
    is_curve : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}$'},
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}_1$'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}_2$'},
    },
    annotate=[
        {'x' : eq_1, 'x_label' : r'$Y^\star_1$', 'y_label' : r'$e^\star_1$', 'func' : is_curve, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y^\star_2$', 'y_label' : r'$e^\star_2$', 'func' : is_curve, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=0.1,
    steps=1000,
    outpath=outpath
)

## Raise in import tariffs
is_curve_1 = lambda x: 2/(x - 11)
is_curve_2 = lambda x: 2/(x - 63)

lm_curve = partial(step_function, center_at=140, value_at_center=0.1)

eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve, 90, 200, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve, 90, 200, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/mf_floating_commercial.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 90, 200, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_2$'},
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'},
    },
    annotate=[
        {'x' : eq_1, 'x_label' : r'$Y^\star_1 = Y^\star_2$', 'y_label' : r'$e^\star_1$', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y^\star_1 = Y^\star_2$', 'y_label' : r'$e^\star_2$', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=0.1,
    steps=1000,
    outpath=outpath
)

# Fix exchange rate
# Change in G
is_curve_1 = lambda x: 2/x
is_curve_2 = lambda x: 4/(2*x-1)

lm_curve_1 = partial(step_function, center_at=0.5, value_at_center=6)
lm_curve_2 = partial(step_function, center_at=1, value_at_center=6)

eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve_1, 0.1, 2, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve_2, 0.9, 2, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/mf_g_fixed.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0.1, 2, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_2$'},
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}_1$'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}_2$'}
    },
    annotate=[
        {'x' : eq_1, 'x_label' : r'$Y^\star_1$', 'y_label' : r'$e^\star$', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y^\star_2$', 'y_label' : r'$e^\star$', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    add_lines = [
        {'coords' : ((0,is_curve_1(eq_1)), (2, is_curve_1(eq_1))),
        'color' : 'black',
        'linestyle' : ':',
        'linewidth' : 1}
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=6,
    steps=1000,
    outpath=outpath
)


# Fix exchange rate
# Depreciation
is_curve_1 = lambda x: 2/(1+x)

lm_curve_1 = partial(step_function, center_at=1/3, value_at_center=6)
lm_curve_2 = partial(step_function, center_at=1, value_at_center=6)


eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve_1, 0.1, 2, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_1, lm_curve_2, 0.1, 2, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/mf_depreciation.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0.1, 2, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}$'},
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}_1$'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}_2$'}
    },
    annotate=[
        {'x' : eq_1, 'x_label' : r'$Y^\star_1$', 'y_label' : r'$e^\star_1$', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y^\star_2$', 'y_label' : r'$e^\star_2$', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    add_lines = [
        {'coords' : ((0,is_curve_1(eq_1)), (2, is_curve_1(eq_1))),
        'color' : 'black',
        'linestyle' : ':',
        'linewidth' : 1},
        {'coords' : ((0, is_curve_1(eq_2)), (2, is_curve_1(eq_2))),
        'color' : 'red',
        'linestyle' : ':',
        'linewidth' : 1},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=2,
    steps=1000,
    outpath=outpath
)


# Fix exchange rate
# Raise in import tariffs
is_curve_1 = lambda x: 2/x
is_curve_2 = lambda x: 4/x

lm_curve_1 = partial(step_function, center_at=1/2, value_at_center=6)
lm_curve_2 = partial(step_function, center_at=1, value_at_center=6)

eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve_1, 0.1, 2, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_1, lm_curve_2, 0.1, 2, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/mf_fixed_commercial.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0.1, 2, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_2$'},
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}_1$'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}_2$'}
    },
    annotate=[
        {'x' : eq_1, 'x_label' : r'$Y^\star_1$', 'y_label' : r'$e^\star$', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y^\star_2$', 'y_label' : r'$e^\star$', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    add_lines = [
        {'coords' : ((0,is_curve_1(eq_1)), (2, is_curve_1(eq_1))),
        'color' : 'black',
        'linestyle' : ':',
        'linewidth' : 1},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=6,
    steps=1000,
    outpath=outpath
)

# Exercises
# 1.1, floating exchange, decrease in C

is_curve_1 = lambda x: 2/x
is_curve_2 = lambda x: 1/x
lm_curve_1 = partial(step_function, center_at=1/2, value_at_center=6)

eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve_1, 0.1, 2, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve_1, 0.1, 2, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/mundell-fleming_ex1_1.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0.1, 2, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'
        },
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_2$'
        },
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'}
    },
    annotate = [
        {'x' : eq_1, 'x_label' : r'$Y^\star_1 = Y^\star_2$', 'y_label' : r'$e^\star_1$', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y^\star_1 = Y^\star_2$', 'y_label' : r'$e^\star_2$', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=6,
    steps=1000,
    outpath=outpath)

# 1.2 floating exchange, increase in imports
is_curve_1 = lambda x: 2/x
is_curve_2 = lambda x: 1/x
lm_curve_1 = partial(step_function, center_at=1/2, value_at_center=6)

eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve_1, 0.1, 2, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve_1, 0.1, 2, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/mundell-fleming_ex1_2.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0.1, 2, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'
        },
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_2$'
        },
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'}
    },
    annotate = [
        {'x' : eq_1, 'x_label' : r'$Y^\star_1 = Y^\star_2$', 'y_label' : r'$e^\star_1$', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y^\star_1 = Y^\star_2$', 'y_label' : r'$e^\star_2$', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=6,
    steps=1000,
    outpath=outpath)


# 2.2, fixed exchange, increase in imports

is_curve_1 = lambda x: 2/x
is_curve_2 = lambda x: 1/x
lm_curve_1 = partial(step_function, center_at=1/2, value_at_center=6)
lm_curve_2 = partial(step_function, center_at=1/4, value_at_center=6)

eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve_1, 0.1, 2, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve_2, 0.1, 2, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/mundell-fleming_ex2_2.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0.1, 2, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'
        },
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_2$'
        },
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}_1$'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}_2$'}
    },
    annotate = [
        {'x' : eq_1, 'x_label' : r'$Y^\star_1$', 'y_label' : r'$e^\star$', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y^\star_2$', 'y_label' : r'$e^\star$', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    add_lines = [
        {'coords' : ((0, is_curve_1(eq_1)), (2, is_curve_1(eq_1))),
        'color' : 'black',
        'linestyle' : ':',
        'linewidth' : 1},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=6,
    steps=1000,
    outpath=outpath)

# 3.1
is_curve = lambda x: (1600-x)/200
lm_curve = partial(step_function, center_at=1200, value_at_center=12)

outpath = os.path.join(base_path, 'static/img/macro_2/mundell-fleming_ex3_1.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 2000, {
    # Functions
    is_curve : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}$'
        },
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'
        }
    },
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=10,
    steps=1000,
    outpath=outpath)

# 3.3
is_curve_1 = lambda x: (1600-x)/200
is_curve_2 = lambda x: (1800-x)/200

lm_curve = partial(step_function, center_at=1200, value_at_center=12)

eq_1 = 1200
eq_2 = 1200

outpath = os.path.join(base_path, 'static/img/macro_2/mundell-fleming_ex3_3.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 2000, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'},
    is_curve_2 : {
    'color' : colors['demand']['shifted'],
    'linestyle' : '-',
    'linewidth' : 4,
    'label' : r'$IS^{MF}_2$'},
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'}
    },
    annotate = [
        {'x' : eq_1, 'x_label' : f'{eq_1:.2f}' , 'y_label' : f'{is_curve_1(eq_1):.2f}', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : f'{eq_2:.2f}', 'y_label' : f'{is_curve_2(eq_2):.2f}', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--' },
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=10,
    steps=1000,
    outpath=outpath)


# 3.4
is_curve_1 = lambda x: (1600-x)/200
is_curve_2 = lambda x: (1800-x)/200

lm_curve_1 = partial(step_function, center_at=1200, value_at_center=12)
lm_curve_2 = partial(step_function, center_at=1400, value_at_center=12)
eq_1 = 1200
eq_2 = 1400

outpath = os.path.join(base_path, 'static/img/macro_2/mundell-fleming_ex3_4.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 2000, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'},
    is_curve_2 : {
    'color' : colors['demand']['shifted'],
    'linestyle' : '-',
    'linewidth' : 4,
    'label' : r'$IS^{MF}_2$'},
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'}
    },
    annotate = [
        {'x' : eq_1, 'x_label' : f'{eq_1:.0f}' , 'y_label' : f'{is_curve_1(eq_1):.2f}', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : f'{eq_2:.0f}', 'y_label' : f'{is_curve_2(eq_2):.2f}', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--' },
    ],
    add_lines = [
        {'coords' : ((0, is_curve_1(eq_1)), (2000, is_curve_1(eq_1))), 
        'color' : 'black', 
        'linestyle' : ':',
        'linewidth' : 1},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=10,
    steps=1000,
    outpath=outpath)


# 5.1
is_curve_1 = lambda x: 5/x
is_curve_2 = lambda x: 4/x

lm_curve_1 = partial(step_function, center_at=1, value_at_center=12)
lm_curve_2 = partial(step_function, center_at=2, value_at_center=12)
eq_1 = 1
eq_2 = 2

outpath = os.path.join(base_path, 'static/img/macro_2/mundell-fleming_ex5_1.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 4, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'},
    is_curve_2 : {
    'color' : colors['demand']['shifted'],
    'linestyle' : '-',
    'linewidth' : 4,
    'label' : r'$IS^{MF}_2$'},
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'}
    },
    annotate = [
        {'x' : eq_1, 'x_label' : f'{eq_1:.0f}' , 'y_label' : f'{is_curve_1(eq_1):.2f}', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : f'{eq_2:.0f}', 'y_label' : f'{is_curve_2(eq_2):.2f}', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--' },
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=10,
    steps=1000,
    outpath=outpath)



# 5.2
is_curve_1 = lambda x: 5/x
is_curve_2 = lambda x: 4/x

lm_curve_1 = partial(step_function, center_at=1, value_at_center=12)
eq_1 = 1

# Compute the money supply that ensures e = 5/1 = 0.2
value_lm_2 = 4 * 0.2
lm_curve_2 = partial(step_function, center_at=value_lm_2, value_at_center=12)
eq_2 = value_lm_2

outpath = os.path.join(base_path, 'static/img/macro_2/mundell-fleming_ex5_2.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 4, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS^{MF}_1$'},
    is_curve_2 : {
    'color' : colors['demand']['shifted'],
    'linestyle' : '-',
    'linewidth' : 4,
    'label' : r'$IS^{MF}_2$'},
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM^{MF}$'}
    },
    annotate = [
        {'x' : eq_1, 'x_label' : f'{eq_1:.0f}' , 'y_label' : f'{is_curve_1(eq_1):.2f}', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : f'{eq_2:.2f}', 'y_label' : f'{is_curve_2(eq_2):.2f}', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--' },
    ],
    add_lines = [
        {'coords' : ((0, is_curve_1(eq_1)), (4, is_curve_1(eq_1))),
        'color' : 'black',
        'linestyle' : ':',
        'linewidth' : 1},
    ],
    x_title=r'$Y$',
    y_title=r'$e$',
    legend=True,
    ymax=10,
    steps=1000,
    outpath=outpath)
