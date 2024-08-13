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

# IS curve
# Original
# Y = 10 + 0.6 (Y-3) + 1/r + 8
# r = 5/(2y-81)
# Increase G by 10
# Y = 10 + 0.6 (Y-3) + 1/r + 8 + 10
# r = 5/(2y-131)

is_curve_1 = lambda x: 5/(2*x - 81)
is_curve_2 = lambda x: 5/(2*x - 131)

# With the change in G, IS moves to the right by
# 1/(1 - c1) delta_g = 2.5 * 10 = 25
change_in_is = 25

outpath = os.path.join(base_path, 'static/img/macro_2/is_g_change_10.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 90, 200, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS curve'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'IS curve, $\Delta G=10$'},
    },
    annotate=[
        {'x' : 120, 'x_label' : '{:.2f}'.format(120), 'y_label' : '{:2f}'.format(is_curve_1(120)), 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 120+change_in_is, 'x_label' : '{:.2f}'.format(120+change_in_is), 'y_label' : '{:2f}'.format(is_curve_2(120+change_in_is)), 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},],
    x_title=r'Y Production',
    y_title=r'r',
    legend=True,
    outpath=outpath
)

# IS curve
# Original
# Y = 10 + 0.6 (Y-3) + 1/r + 8
# r = 5/(2y-81)
# Increase G by 10
# Y = 10 + 0.6 (Y-3) + 1/r + 8 + 10
# r = 5/(2y-131)
# LM curve
# 457477 = Y^2/r
# r = y^2/457477

is_curve_1 = lambda x: 5/(2*x - 81)
is_curve_2 = lambda x: 5/(2*x - 131)
lm_curve = lambda x: (x**2)/457477

# With the change in G, IS moves to the right by
# 1/(1 - c1) delta_g = 2.5 * 10 = 25
change_in_is = 25

eq = find_equilibrium_in_range(is_curve_2, lm_curve, 70, 200)

outpath = os.path.join(base_path, 'static/img/macro_2/is-lm_g_change_10.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 70, 200, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS curve'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'IS curve, $\Delta G=10$'},
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM curve'},
    },
    annotate=[
        {'x' : 120, 'x_label' : r'$Y^\star_1$', 'y_label' : r'$r^\star_1$', 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 120+change_in_is, 'x_label' : '{:.2f}'.format(120+change_in_is), 'y_label' : r'$r^\star_1$', 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq, 'x_label' : r'$Y^\star_2$', 'y_label' : r'$r^\star_2$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'}],
    x_title=r'Y Production',
    y_title=r'r',
    legend=True,
    ymax=0.1,
    outpath=outpath
)

# IS vertical
change_in_is = 25

is_curve_1 = partial(step_function, center_at=120, value_at_center=1) 
is_curve_2 = partial(step_function, center_at=120+change_in_is, value_at_center=1)
lm_curve = lambda x: (x**2)/457477

eq_2 = find_equilibrium_in_range(is_curve_1, lm_curve, 70, 200)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve, 70, 200)

outpath = os.path.join(base_path, 'static/img/macro_2/is-vertical_change_g.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 70, 200, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS curve'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'IS curve, $\Delta G=10$'},
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM curve'},
    },
    annotate=[
        {'x' : 120, 'x_label' : r'$Y^\star_1 (120)$', 'y_label' : r'$r^\star_1$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 120+change_in_is, 'x_label' : r'$Y^\star_2 (145)$', 'y_label' : r'$r^\star_2$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},],
    x_title=r'Y Production',
    y_title=r'r',
    legend=True,
    ymax=0.1,
    outpath=outpath
)

# LM horizontal
change_in_is = 25

is_curve_1 = lambda x: 5/(2*x - 81)
is_curve_2 = lambda x: 5/(2*x - 131)
lm_curve = lambda x: is_curve_1(120) + 0.00000001 * x

eq_2 = find_equilibrium_in_range(is_curve_1, lm_curve, 70, 200)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve, 70, 200)

outpath = os.path.join(base_path, 'static/img/macro_2/lm-horizontal_change_g.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 70, 200, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS curve'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'IS curve, $\Delta G=10$'},
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM curve'},
    },
    annotate=[
        {'x' : 120, 'x_label' : r'$Y^\star_1 (120)$', 'y_label' : r'$r^\star_1, r^\star_2$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 120+change_in_is, 'x_label' : r'$Y^\star_2 (145)$', 'y_label' : '', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},],
    x_title=r'Y Production',
    y_title=r'r',
    legend=True,
    ymax=0.1,
    outpath=outpath
)

# Change in T
# IS curve
# Original
# Y = 10 + 0.6 (Y-3) + 1/r + 8
# r = 5/(2y-81)
# Increase G by 10
# Y = 10 + 0.6 (Y+7) + 1/r + 8
# r = 5/(2y-111)
# LM curve
# 457477 = Y^2/r
# r = y^2/457477

is_curve_1 = lambda x: 5/(2*x - 81)
is_curve_2 = lambda x: 5/(2*x - 111)
lm_curve = lambda x: (x**2)/457477

change_in_is = 0.6/(1-0.6)*10

eq = find_equilibrium_in_range(is_curve_2, lm_curve, 70, 200)

outpath = os.path.join(base_path, 'static/img/macro_2/is-change-t.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 70, 200, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS curve'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'IS curve, $\Delta T=-10$'},
    },
    annotate=[
        {'x' : 120, 'x_label' : '{:.2f}'.format(120), 'y_label' : '{:.2f}'.format(lm_curve(120)), 'func' : is_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 120+change_in_is, 'x_label' : '{:.2f}'.format(120+change_in_is), 'y_label' : '{:2f}'.format(is_curve_2(120+change_in_is)), 'func' : is_curve_2, 'color' : colors['other'], 'linestyle' : '--'},],
    x_title=r'Y Production',
    y_title=r'r',
    legend=True,
    ymax=0.1,
    outpath=outpath
)

# Change in M
lm_curve_1 = lambda x: x**1.2 /100
lm_curve_2 = lambda x: x**1.2 /200

x_val = 20 

outpath = os.path.join(base_path, 'static/img/macro_2/lm-change-m.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 10, 30, {
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
        'label' : r'LM, $\Delta \frac{M^s}{p}=100$'},
    },
    annotate=[
        {'x' : x_val, 'x_label' : '{:.2f}'.format(x_val), 'y_label' : '{:.4f}'.format(lm_curve_1(x_val)), 'func' : lm_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : x_val, 'x_label' : '{:.2f}'.format(x_val), 'y_label' : '{:.4f}'.format(lm_curve_2(x_val)), 'func' : lm_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
        ],
    x_title=r'Y Production',
    y_title=r'r',
    legend=True,
    outpath=outpath
)



# Change in M
lm_curve_1 = lambda x: x**1.2 /100
lm_curve_2 = lambda x: x**1.2 /200
is_curve = lambda x: 5/(2*x-13)

x_val = 20 
eq_1 = find_equilibrium_in_range(lm_curve_1, is_curve, 10, 30)
eq_2 = find_equilibrium_in_range(lm_curve_2, is_curve, 10, 30)

outpath = os.path.join(base_path, 'static/img/macro_2/is-lm-change-m.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 10, 30, {
    # Functions
    is_curve : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'IS'},
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'LM, $\Delta \frac{M^s}{p}=100$'},
    },
    annotate=[
        {'x' : eq_1, 'x_label' : r'$Y^\star_1$', 'y_label' : r'$r^\star_1$', 'func' : lm_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y^\star_2$', 'y_label' : r'$r^\star_2$', 'func' : lm_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    x_title=r'Y Production',
    y_title=r'r',
    legend=True,
    outpath=outpath
)

# Liquidity trap
# Creating the LM curve
# Money demand
def money_demand_r(m, y,at_zero=10):
    t = 2/m+y-at_zero
    
    # Replace negative values for 0
    return np.where(t < 0, 0, t)

at_zero = 10
m_demand_0 = partial(money_demand_r, y=6, at_zero=at_zero)
m_demand_1 = partial(money_demand_r, y=8, at_zero=at_zero)
m_demand_2 = partial(money_demand_r, y=9, at_zero=at_zero)

m_supply = partial(step_function, center_at=0.5, value_at_center=20)


# LM curve
m_supply_exo = 0.5
def lm_curve(y, ms = m_supply_exo):
    t = 2/0.5 + y - at_zero

    return np.where(t < 0, 0, t)


outpath = os.path.join(base_path, 'static/img/macro_2/liquidity-trap.webp')
fig, ax = plt.subplots(1, 2, sharex=False, sharey=True)

plot_multiple_functions(ax[0], 0.1, 2.5, {
    # Functions
    m_demand_0 : {
        'color' : colors['demand']['normal'],
        'apha' : 1,
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM',
        'label' : r'$M^d, Y_0=6$',
        },
    m_demand_1 : {
        'color' : colors['demand']['normal'],
        'alpha' : 0.5,
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM',
        'label' : r'$M^d, Y_1=8$',
        },
    m_demand_2 : {
        'color' : colors['demand']['normal'],
        'alpha' : 0.25,
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM',
        'label' : r'$M^d, Y_2=9$',},
    m_supply : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$M^s$',
        }
    },
    annotate=[
        {'x' : 0.5, 'x_label' : r'$\frac{M}{p}$', 'y_label' : r'$r_0$', 'func' : m_demand_0, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 0.5, 'x_label' : r'$\frac{M}{p}$', 'y_label' : r'$r_1$', 'func' : m_demand_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 0.5, 'x_label' : r'$\frac{M}{p}$', 'y_label' : r'$r_2$', 'func' : m_demand_2, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    x_title = r'$\frac{M}{p}$',
    y_title = r'$r$',
    ymax = 4,
    ymin = 0,
    steps = 1000,
    legend=True,
    outpath=outpath
)
plot_multiple_functions(ax[1], 5, 10, {
    # Functions
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM'},
    },
    annotate=[
        {'x' : 6, 'x_label' : r'$Y=6$', 'y_label' : r'$r_0$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 8, 'x_label' : r'$Y=8$', 'y_label' : r'$r_1$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : 9, 'x_label' : r'$Y=9$', 'y_label' : r'$r_2$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    x_title = r'$Y$',
    y_title = r'$r$',
    ymax = 4,
    steps = 1000,
    legend=True,
    outpath=outpath
)

# Great Depression
def is_curve_examples(y, alpha=1, c0=1, c1=0.5, g=0, t=0):
    return alpha/(y*(1-c1)-c0-g+c1*t)

is_curve_0 = partial(is_curve_examples, alpha = 1, c0 = 2, c1 = 0.6, g = 4, t = 4)
is_curve_1 = partial(is_curve_examples, alpha = 0.5, c0 = 0.5, c1 = 0.6, g = 4, t = 4)
is_curve_2 = partial(is_curve_examples, alpha = 0.5, c0 = 0.5, c1 = 0.6, g = 6, t = 4)

lm_curve = lambda x: (x-15.5)**0.8

eq_0 = find_equilibrium_in_range(is_curve_0, lm_curve, 15.5, 20, steps=1000)
eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve, 15.5, 20, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve, 15.5, 20, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/great_depression.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 15.5, 17, {
    # Functions
    is_curve_0 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_0$'},
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'alpha' : 0.5,
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_1$'},
    is_curve_2 : {
        'color' : colors['demand']['normal'],
        'alpha' : 0.25,
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_2$'},
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM'},
    },
    annotate=[
        {'x' : eq_0, 'x_label' : r'$Y_0$', 'y_label' : r'$r_0$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_1, 'x_label' : r'$Y_1$', 'y_label' : r'$r_1$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y_2$', 'y_label' : r'$r_2$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    x_title = r'$Y$',
    y_title = r'$r$',
    ymax = 0.5,
    legend=True,
    steps=1000,
    outpath=outpath
)

# Ex 2

is_curve_1 = lambda x :  (-1) * (0.35 * x- 4)/2
is_curve_2 = lambda x :  (-1) * (0.35 * x- 8)/2
lm_curve = lambda x: (4*x-2)/2

outpath = os.path.join(base_path, 'static/img/macro_2/ex_2.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 3, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_1$'},
    is_curve_2 : {
        'color' : colors['demand']['normal'],
        'alpha' : 0.5,
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_2$'},
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM'},
    },
    annotate=[],
    x_title = r'$Y$',
    y_title = r'$r$',
    legend=True,
    outpath=outpath
)

# Ex 3
# Part 1, IS shocks

is_curve_1 = lambda x :  (-1) * (0.35 * x- 4)/2
is_curve_2 = lambda x :  (-1) * (0.35 * x- 8)/2
lm_curve_1 = lambda x: 1
lm_curve_2 = lambda x: (0.2*x-0.5)

eq_1_1 = find_equilibrium_in_range(is_curve_1, lm_curve_1, 0, 20, steps=1000)
eq_1_2 = find_equilibrium_in_range(is_curve_1, lm_curve_2, 0, 20, steps=1000)
eq_2_1 = find_equilibrium_in_range(is_curve_2, lm_curve_1, 0, 20, steps=1000)
eq_2_2 = find_equilibrium_in_range(is_curve_2, lm_curve_2, 0, 20, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/ex_3_1.png')
fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(10, 5))

plot_multiple_functions(ax[0], 0, 20, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_1$'},
    is_curve_2 : {
        'color' : colors['demand']['normal'],
        'alpha' : 0.5,
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_2$'},
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM, r fixed'},
    },
    annotate=[
        {'x' : eq_1_1, 'x_label' : r'$Y_1$', 'y_label' : r'$r_1 = 1$', 'func' : lm_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2_1, 'x_label' : r'$Y_2$', 'y_label' : r'$r_1 = 1$', 'func' : lm_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    ymax=4,
    x_title = r'$Y$',
    y_title = r'$r$',
    legend=True,
    outpath=outpath
)

plot_multiple_functions(ax[1], 0, 20, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_1$'},
    is_curve_2 : {
        'color' : colors['demand']['normal'],
        'alpha' : 0.5,
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_2$'},
    lm_curve_2 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM, r changes'},
    },
    annotate=[
        {'x' : eq_1_2, 'x_label' : r'$Y_1$', 'y_label' : fr'$r_2 = {is_curve_1(eq_1_2):.2f}$', 'func' : lm_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2_2, 'x_label' : r'$Y_2$', 'y_label' : fr'$r_2 = {is_curve_2(eq_2_2):.2f}$', 'func' : lm_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
    ],
    ymax=4,
    x_title = r'$Y$',
    y_title = r'$r$',
    legend=True,
    outpath=outpath
)


# Ex. 4

# IS curve
# Original
# Y = 10 + 0.6 (Y-3) + 1/r + 8
# r = 5/(2y-81)
# Increase G by 10
# Y = 10 + 0.6 (Y-3) + 1/r + 8 + 10
# r = 5/(2y-131)

is_curve_1 = lambda x: 5/(2*x - 81)
is_curve_2 = lambda x: 5/(2*x - 131)
lm_curve = lambda x: x**0.7 / 10**3

eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve, 90, 200, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve, 90, 200, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/ex_4_1.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 90, 200, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_1$'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_2$'},
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : 'LM'},
    },
    annotate=[
        {'x' : eq_1, 'x_label' : r'$Y_1$', 'y_label' : r'$r_1$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y_2$', 'y_label' : r'$r_2$', 'func' : lm_curve, 'color' : colors['other'], 'linestyle' : '--'},
        ],
    x_title=r'Y',
    y_title=r'r',
    legend=True,
    outpath=outpath
)

# LM depends on Y-T
# A cut in T

is_curve_1 = lambda x: 896/(4279+448*x)
is_curve_2 = lambda x: 448/(224*x-521)
lm_curve_1 = lambda x: (448*x-9655)/44800
lm_curve_2 = lambda x: (224*x-2167)/22400

eq_1 = find_equilibrium_in_range(is_curve_1, lm_curve_1, 18, 30, steps=1000)
eq_2 = find_equilibrium_in_range(is_curve_2, lm_curve_2, 18, 30, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2/ex_4_2.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 14, 30, {
    # Functions
    is_curve_1 : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_1$'},
    is_curve_2 : {
        'color' : colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS_2$'},
    lm_curve_1 : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM_1$'},
    lm_curve_2 : {
        'color' : colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM_2$'},
    },
    annotate=[
        {'x' : eq_1, 'x_label' : r'$Y_1$', 'y_label' : r'$r_1$', 'func' : lm_curve_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_2, 'x_label' : r'$Y_2$', 'y_label' : r'$r_2$', 'func' : lm_curve_2, 'color' : colors['other'], 'linestyle' : '--'},
        ],
    x_title=r'Y',
    y_title=r'r',
    legend=True,
    outpath=outpath
)

# Ex 6
# 6.1
is_curve = lambda x: 15 - 0.005*x

outpath = os.path.join(base_path, 'static/img/macro_2/ex_6_1.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 1400, {
    # Functions
    is_curve : {
        'color' : colors['demand']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$IS$'},
    },
    annotate=[],
    x_title=r'Y',
    y_title=r'r',
    legend=True,
    ymin=8,
    outpath=outpath
)

# 6.2
lm_curve = lambda x: (x-1000)/200

outpath = os.path.join(base_path, 'static/img/macro_2/ex_6_2.png')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 1400, {
    # Functions
    lm_curve : {
        'color' : colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r'$LM$'},
    },
    annotate=[],
    x_title=r'Y',
    y_title=r'r',
    legend=True,
    outpath=outpath
)
