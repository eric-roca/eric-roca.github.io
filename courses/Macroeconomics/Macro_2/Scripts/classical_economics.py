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

    # Sometimes, the function ws pass are constants
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



def plot_multiple_functions(ax, xmin, xmax, funcs, annotate=None, x_title='x', y_title='y', legend=False, steps=100, ymax=None, reverse_axes = False, outpath=None):
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
}



production = lambda x : x**0.5

vals_to_annotate = [1, 2, 6, 7]
annotate = [{'x' : i, 'x_label' : r'$x_{{{}}}$'.format(enu), 'y_label' : r'$y_{{{}}}$'.format(enu), 'func' : production, 'color' : colors['other'], 'linestyle' : '--'} for enu, i in enumerate(vals_to_annotate)]

outpath = os.path.join(base_path, 'static/img/macro_2', 'production_function.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 10, {
    #Functions
    production : {
        'color' : colors['demand']['normal'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r"$f(x)$: Production"}
    },
    annotate=annotate, legend=True, outpath=outpath)

# Labour demand
x_max = 10
labour_demand = lambda x: x**(-0.5)
vals_to_annotate = [2]
annotate = [{'x' : i, 'x_label' : r'$N^\star$', 'y_label' : r'$\left(\frac{w}{p}\right)^\star$', 'func' : labour_demand, 'color' : colors['other'], 'linestyle' : '--', 'x_max' : x_max} for enu, i in enumerate(vals_to_annotate)]

outpath = os.path.join(base_path, 'static/img/macro_2', 'labour_demand_function.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, x_max, {
    #Functions
    labour_demand : {
        'color' : colors['demand']['normal'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r"$f(x)$: Production"
        }
    },
    annotate=annotate, 
    x_title=r'$N$ Nombre de travailleurs', 
    y_title=r'$\frac{w}{p}$ Salaire', 
    legend=True, outpath=outpath)

# Changes in labour supply
labour_demand = lambda x: 150-x**0.9
labour_supply = lambda x: x**1.2
labour_supply_reduced = lambda x: x**1.2 + 100
eq_1 = find_equilibrium_in_range(labour_demand, labour_supply, 0, 100)
eq_2 = find_equilibrium_in_range(labour_demand, labour_supply_reduced, 0, 100)

outpath = os.path.join(base_path, 'static/img/macro_2', 'classical_change_labour_eq.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 100, 
    # Functions
    {labour_demand: 
        {'color': colors['demand']['normal'], 
        'linestyle': '-', 
        'linewidth' : 4,
        'label' : r'$N^d$: Demande travail'
        }, 
    labour_supply: 
        {'color': colors['supply']['normal'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r'$N^s_0$: Offre travail initiale'
        },
    labour_supply_reduced: 
        {'color': colors['supply']['shifted'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r'$N^s_1$: Offre travail modifié'}}, 
    annotate=[
        {'x': eq_1, 'x_label': r'$N^\star_1$', 'y_label': r'$\left(\frac{w}{p}\right)^\star_1$', 'func': labour_demand, 'color': colors['other'], 'linestyle': '--'},
        {'x': eq_2, 'x_label': r'$N^\star_2$', 'y_label': r'$\left(\frac{w}{p}\right)^\star_2$', 'func': labour_demand, 'color': colors['other'], 'linestyle': '--'},],
    x_title=r'$N$ Travailleurs', 
    y_title=r'$\frac{w}{p}$ Salaire réel', 
    legend=True, outpath=outpath)


# Money demand
money_demand = lambda x: 1/x**0.5
money_supply = partial(step_function, center_at=2, value_at_center=10)

eq_money = find_equilibrium_in_range(money_demand, money_supply, 0, 3, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2', 'eq_money.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 3, 
    # Functions
    {money_demand: 
        {'color': colors['demand']['normal'], 
        'linestyle': '-', 
        'linewidth' : 4,
        'label' : r"$M^d$: Demande d'encaisses"
        },
    money_supply: 
        {'color': colors['supply']['normal'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r"$\bar{Y}$: Niveau de production"
        }
    },
    annotate=[
        {'x' : eq_money, 'x_label' : r'$\bar{Y}$', 'y_label' : r'$p^\star$', 'func' : money_demand, 'color' : 'black', 'linestyle' : '--'}],
    x_title=r'$Y$ Production', 
    y_title=r'$p$ Prix', 
    legend=True, 
    ymax=3, steps=1000, outpath=outpath)

del(money_demand, money_supply, eq_money)


# Change in money supply
money_demand_1 = lambda x: 1/x**0.5
money_demand_2 = lambda x: 1.5/x**0.5
money_supply= partial(step_function, center_at=2, value_at_center=10)

eq_money_1 = find_equilibrium_in_range(money_demand_1, money_supply, 0, 3, steps=10000)
eq_money_2 = find_equilibrium_in_range(money_demand_2, money_supply, 0, 3, steps=10000)

outpath = os.path.join(base_path, 'static/img/macro_2', 'classical_change_money_eq.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 3, 
    # Functions
    {money_demand_1: 
        {'color': colors['demand']['normal'], 
        'linestyle': '-', 
        'linewidth' : 4,
        'label' : r"$M^d_1$: Demande d'encaisses"
        },
    money_demand_2: 
        {'color': colors['demand']['shifted'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r"$M^d_2$: Demande d'encaisses modifié"
        },
    money_supply: 
        {'color': colors['supply']['normal'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r"$\bar{Y}$: Niveau de production"
        }
    },
    annotate=[
        {'x' : eq_money_1, 'x_label' : r'$\bar{Y}$', 'y_label' : r'$p^\star_1$', 'func' : money_demand_1, 'color' : 'black', 'linestyle' : '--'},
        {'x' : eq_money_2, 'x_label' : r'$\bar{Y}$', 'y_label' : r'$p^\star_2$', 'func' : money_demand_2, 'color' : 'black', 'linestyle' : '--'}],
    x_title=r'$Y$ Production', 
    y_title=r'$p$ Prix', 
    legend=True, 
    ymax=3, steps=10000, outpath=outpath)

# Interest rate
savings_supply = partial(step_function, center_at=2, value_at_center=10)
savings_demand = lambda x: 1.2/x**0.6

eq_savings = find_equilibrium_in_range(savings_demand, savings_supply, 0, 3, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2', 'eq_savings.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 3, 
    # Functions
    {savings_demand: 
        {'color': colors['demand']['normal'], 
        'linestyle': '-',
        'linewidth' : 4,
        'label' : r"$I(r)$: Demande d'épargne"
        },
    savings_supply: 
        {'color': colors['supply']['normal'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r"$S$: Offre d'épargne"
        }
    },
    annotate=[
        {'x' : eq_savings, 'x_label' : r'$\bar{S}$', 'y_label' : r'$r^\star$', 'func' : savings_demand, 'color' : colors['other'], 'linestyle' : '--'}],
    x_title=r'$S$ Épargne', 
    y_title=r"$r$ Taux d'interet", 
    legend=True, ymax=3, steps=1000, outpath=outpath)

# Change in savings
# Interest rate
savings_supply_1 = partial(step_function, center_at=2, value_at_center=10)
savings_supply_2 = partial(step_function, center_at=1.3, value_at_center=10)
savings_demand = lambda x: 1.2/x**0.6

eq_savings_1 = find_equilibrium_in_range(savings_demand, savings_supply_1, 0, 3, steps=1000)
eq_savings_2 = find_equilibrium_in_range(savings_demand, savings_supply_2, 0, 3, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2', 'change_savings_eq.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 3, 
    # Functions
    {savings_demand: 
        {'color': colors['demand']['normal'], 
        'linestyle': '-',
        'linewidth' : 4,
        'label' : r"$I(r)$: Demande d'épargne"
        },
    savings_supply_1: 
        {'color': colors['supply']['normal'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r"$S_2$: Offre d'épargne"
        },
    savings_supply_2:
        {'color': colors['supply']['shifted'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r"$S_2$: Offre d'épargne modifié"
        }
    },
    annotate=[
        {'x' : eq_savings_1, 'x_label' : r'$\bar{S}_1$', 'y_label' : r'$r^\star_1$', 'func' : savings_demand, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_savings_2, 'x_label' : r'$\bar{S}_2$', 'y_label' : r'$r^\star_2$', 'func' : savings_demand, 'color' : colors['other'], 'linestyle' : '--'}],
    x_title=r'$S$ Épargne', 
    y_title=r"$r$ Taux d'interet", 
    legend=True, ymax=3, steps=1000, outpath=outpath)


# All together

# Functions
labour_demand = lambda x: x**(-0.5)
labour_supply = lambda x: x**0.5
production = lambda x: 4 * x**0.7
fixed_labour = partial(step_function, center_at=1, value_at_center=20)
production_fixed = partial(step_function, center_at=4, value_at_center=20)
money_demand = lambda x: 4/x**0.5
xmax = 14
ymax_production = xmax
# Partials

plt_labour = partial(plot_multiple_functions,
    xmin=0, xmax=xmax,
    funcs={labour_demand:
        {'color': colors['demand']['normal'],
        'linestyle': '-',
        'linewidth' : 4,
        'label' : r"$N^d$: Demande travail"
        },
    labour_supply:
        {'color': colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r"$N^s$: Offre travail"
        }
    },
    annotate=[
        {'x': 1, 'x_label': r'$N^\star$', 'y_label': r'$\left(\frac{w}{p}\right)^\star$', 'func': labour_demand, 'color': colors['other'], 'linestyle': '--'},],
        x_title=None,
        y_title=None,
        reverse_axes=True
)

plt_production = partial(plot_multiple_functions,
    xmin=0, xmax=xmax,
    funcs={production:
        {'color': colors['demand']['normal'],
        'linestyle': '-',
        'linewidth' : 4,
        'label' : r"$f(x)$: Production"
        },
    fixed_labour:
        {'color': colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r"$N^s$: Offre travail"
        }
    },
    annotate=[
        {'x': 1, 'x_label': r'$N^\star$', 'y_label': r'$\bar{Y}$', 'func': production, 'color': colors['other'], 'linestyle': '--'},],
        x_title=None,
        y_title=None,
        steps = 1000,
        ymax = ymax_production,
        reverse_axes=True
)

plt_money = partial(plot_multiple_functions,
    xmin=0, xmax=ymax_production,
    funcs={money_demand:
        {'color': colors['demand']['normal'],
        'linestyle': '-',
        'linewidth' : 4,
        'label' : r"$M^d$: Demande d'encaisses"
        },
    production_fixed:
        {'color': colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r"$f(x)$: Production"
        }
    },
    annotate=[
        {'x': 4, 'x_label': r'$\bar{Y}$', 'y_label': r'$p^\star$', 'func': money_demand, 'color': colors['other'], 'linestyle': '--'},],
        x_title=None,
        y_title=None,
        steps = 1000,
        ymax = 20,
        reverse_axes=False
)

plts = [
    {'plt' : None},
    {'plt' : plt_money, 'inverse_x' : False, 'inverse_y' : False},
    {'plt' : plt_labour, 'inverse_x' : True, 'inverse_y' : True},
    {'plt' : plt_production, 'inverse_x' : False, 'inverse_y' : True}
]

fig = plt.figure()
gs = plt.GridSpec(2, 2, figure=fig, hspace=0, wspace=0)

outpath = os.path.join(base_path, 'static/img/macro_2', 'all_markets.webp')
for row in range(2):
    for col in range(2):
        if plts[row * 2 + col]['plt'] is None:
            continue
        ax = fig.add_subplot(gs[row, col])
        plts[row * 2 + col]['plt'](ax)
        if plts[row * 2 + col]['inverse_x']:
            ax.invert_xaxis()
        if plts[row * 2 + col]['inverse_y']:
            ax.invert_yaxis()

fig.savefig(outpath, bbox_inches='tight')


# Exercises

# Exercise 1, effects of an increase in K on labour in equilibrium and production

production_1 = lambda x : 2 * x**0.7
production_2 = lambda x : 4 * x**0.7
labour_supply = lambda x: x**1.2
labour_demand_1 = lambda x: 2 * 0.7 * x**(-0.3)
labour_demand_2 = lambda x: 4 * 0.7 * x**(-0.3)
labour_eq_1 = find_equilibrium_in_range(labour_demand_1, labour_supply, 0, 5)
labour_eq_2 = find_equilibrium_in_range(labour_demand_2, labour_supply, 0, 5)
constant_labour_1 = partial(step_function, center_at=labour_eq_1, value_at_center=20)
constant_labour_2 = partial(step_function, center_at=labour_eq_2, value_at_center=20)
xmax = 5

plt_production = partial(plot_multiple_functions,
    xmin=0, xmax=xmax,
    # Functions
    funcs={production_1:
        {'color': colors['demand']['normal'],
        'linestyle': '-',
        'linewidth' : 4,
        'label' : r"$f(x)$: Production 1"
        },
    production_2:
        {'color': colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r"$f(x)$: Production 2"
        },
    constant_labour_1:
        {'color': colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r"$N^\star_1$: Niveau d'emploi 1"
        },
    constant_labour_2:
        {'color': colors['supply']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r"$N^\star_2$: Niveau d'emploi 2"
        }
    },
    annotate=[
        {'x' : labour_eq_1, 'x_label' : r'$N^\star_1$', 'y_label' : r'$Y_1$', 'func' : production_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : labour_eq_2, 'x_label' : r'$N^\star_2$', 'y_label' : r'$Y_2$', 'func' : production_2, 'color' : colors['other'], 'linestyle' : '--'}],
        x_title=None,
        y_title=None,
        steps = 1000,
        legend=True
)

plt_labour = partial(plot_multiple_functions,
    xmin=0, xmax=xmax,
    funcs={labour_demand_1:
        {'color': colors['demand']['normal'],
        'linestyle': '-',
        'linewidth' : 4,
        'label' : r"$N^d_1$: Demande travail 1"
        },
    labour_demand_2:
        {'color': colors['demand']['shifted'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r"$N^d_2$: Demande travail 2"
        },
    labour_supply:
        {'color': colors['supply']['normal'],
        'linestyle' : '-',
        'linewidth' : 4,
        'label' : r"$N^\star_1$: Niveau d'emploi 1"
        },
    },
    annotate=[
        {'x' : labour_eq_1, 'x_label' : r'$N^\star_1$', 'y_label' : r'$\left(\frac{w}{p}\right)^\star_1$', 'func' : labour_demand_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : labour_eq_2, 'x_label' : r'$N^\star_2$', 'y_label' : r'$\left(\frac{w}{p}\right)^\star_2$', 'func' : labour_demand_2, 'color' : colors['other'], 'linestyle' : '--'}],
        x_title=None,
        y_title=None,
        legend=True
)

plts = [
    {'plt' : plt_labour, 'inverse_x' : False, 'inverse_y' : False},
    {'plt' : plt_production, 'inverse_x' : False, 'inverse_y' : True}
]

fig = plt.figure()
gs = plt.GridSpec(2, 1, figure=fig, hspace=0, wspace=0)

outpath = os.path.join(base_path, 'static/img/macro_2', 'classic_ex_1.webp')
for i, p in enumerate(plts):
    ax = fig.add_subplot(gs[i])
    p['plt'](ax)
    if p['inverse_x']:
        ax.invert_xaxis()
    if p['inverse_y']:
        ax.invert_yaxis()

fig.savefig(outpath, bbox_inches='tight')

# Exercise 2, continuation to the savings market

savings_1 = partial(step_function, center_at=2, value_at_center=10)
savings_2 = partial(step_function, center_at=4, value_at_center=10)
demand_savings = lambda x: 1.2/x**0.6

eq_interest_1 = find_equilibrium_in_range(demand_savings, savings_1, 0, 5, steps=1000)
eq_interest_2 = find_equilibrium_in_range(demand_savings, savings_2, 0, 5, steps=1000)

outpath = os.path.join(base_path, 'static/img/macro_2', 'classic_ex_2.webp')
fig, ax = plt.subplots()
plot_multiple_functions(ax, 0, 5,
    # Functions
    {savings_1: 
        {'color': colors['supply']['normal'], 
        'linestyle': '-',
        'linewidth' : 4,
        'label' : r"$S_1$: Offre d'épargne 1"
        },
    savings_2: 
        {'color': colors['supply']['shifted'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r"$S_2$: Offre d'épargne 2"
        },
    demand_savings:
        {'color': colors['demand']['normal'], 
        'linestyle' : '-', 
        'linewidth' : 4,
        'label' : r"$I(r)$: Demande d'épargne"
        }
    },
    annotate=[
        {'x' : eq_interest_1, 'x_label' : r'$S_1$', 'y_label' : r'$r^\star_1$', 'func' : savings_1, 'color' : colors['other'], 'linestyle' : '--'},
        {'x' : eq_interest_2, 'x_label' : r'$S_2$', 'y_label' : r'$r^\star_2$', 'func' : savings_2, 'color' : colors['other'], 'linestyle' : '--'}],
    x_title=r'$S$ Épargne', 
    y_title=r"$r$ Taux d'intérêt", 
    legend=True,
    steps=1000,
    ymax=5)

fig.savefig(outpath, bbox_inches='tight')