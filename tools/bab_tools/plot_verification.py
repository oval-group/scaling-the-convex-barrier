import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.titlepad'] = 10
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import itertools
import os


def plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, fig_name, title,
                            linestyles=None, one_vs_all=False):

    mpl.rcParams['font.size'] = 12
    tables = []
    for filename in file_list:
        m = pd.read_pickle(folder + filename).dropna(how="all")
        tables.append(m)
    # keep only the properties in common
    for m in tables:
        if not one_vs_all:
            m["unique_id"] = m["Idx"].map(int).map(str) + "_" + m["prop"].map(int).map(str)
        else:
            m["unique_id"] = m["Idx"].map(int).map(str)
        m["Eps"] = m["Idx"].map(float)
    for m1, m2 in itertools.product(tables, tables):
        m1.drop(m1[(~m1['unique_id'].isin(m2["unique_id"]))].index, inplace=True)

    timings = []
    for idx in range(len(tables)):
        timings.append([])
        for i in tables[idx][time_name_list[idx]].values:
            if i >= timeout:
                timings[-1].append(float('inf'))
            else:
                timings[-1].append(i)
        timings[-1].sort()
    # check that they have the same length.
    for m1, m2 in itertools.product(timings, timings):
        assert len(m1) == len(m2)
    print(len(m1))

    starting_point = timings[0][0]
    for timing in timings:
        starting_point = min(starting_point, timing[0])

    fig = plt.figure(figsize=(6, 6))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.axhline(linewidth=3.0, y=100, linestyle='dashed', color='grey')

    y_min = 0
    y_max = 100
    ax_value.set_ylim([y_min, y_max + 5])

    min_solve = float('inf')
    max_solve = float('-inf')
    for timing in timings:
        min_solve = min(min_solve, min(timing))
        finite_vals = [val for val in timing if val != float('inf')]
        if len(finite_vals) > 0:
            max_solve = max(max_solve, max([val for val in timing if val != float('inf')]))

    axis_min = starting_point
    axis_min = min(0.5 * min_solve, 1)
    ax_value.set_xlim([axis_min, timeout + 1])

    for idx, (clabel, timing) in enumerate(zip(labels, timings)):
        # Make it an actual cactus plot
        xs = [axis_min]
        ys = [y_min]
        prev_y = 0
        for i, x in enumerate(timing):
            if x <= timeout:
                # Add the point just before the rise
                xs.append(x)
                ys.append(prev_y)
                # Add the new point after the rise, if it's in the plot
                xs.append(x)
                new_y = 100 * (i + 1) / len(timing)
                ys.append(new_y)
                prev_y = new_y
        # Add a point at the end to make the graph go the end
        xs.append(timeout)
        ys.append(prev_y)

        linestyle = linestyles[idx] if linestyles is not None else "solid"
        ax_value.plot(xs, ys, color=colors[idx], linestyle=linestyle, label=clabel, linewidth=3.0)

    ax_value.set_ylabel("% of properties verified", fontsize=15)
    ax_value.set_xlabel("Computation time [s]", fontsize=15)
    plt.xscale('log', nonposx='clip')
    ax_value.legend(fontsize=9.5)
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()

    figures_path = "./plots/"
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    plt.savefig(figures_path + fig_name, format='pdf', dpi=300)


def to_latex_table(folder, bases, file_list, time_name_list, labels, plot_names, timeout, one_vs_all=False):

    # latex_tables
    latex_tables = []
    for base, plt_name in zip(bases, plot_names):
        # Create latex table.
        tables = []
        for filename in file_list:
            m = pd.read_pickle(folder + f"{base}_" + filename).dropna(how="all")
            tables.append(m)
        # keep only the properties in common
        for m in tables:
            if not one_vs_all:
                m["unique_id"] = m["Idx"].map(int).map(str) + "_" + m["prop"].map(int).map(str)
            else:
                m["unique_id"] = m["Idx"].map(int).map(str)
            m["Eps"] = m["Idx"].map(float)
        for m1, m2 in itertools.product(tables, tables):
            m1.drop(m1[(~m1['unique_id'].isin(m2["unique_id"]))].index, inplace=True)

        # Set all timeouts to <timeout> seconds.
        for table, time_name in zip(tables, time_name_list):
            for column in table:
                if "SAT" in column:
                    table.loc[table[column] == 'timeout', time_name] = timeout

        full_table = tables[0]
        for c_table in tables[1:]:
            full_table = pd.merge(full_table, c_table, on=['Idx', 'prop', 'Eps', 'unique_id'], how='inner')

        # Create summary table.
        summary_dict = {}
        for column in full_table:
            if column not in ['Idx', 'prop', 'Eps', 'unique_id']:
                if "SAT" not in column:
                    c_mean = full_table[column].mean()
                    summary_dict[column] = c_mean
                else:
                    # Handle SAT status
                    n_timeouts = len(full_table.loc[full_table[column] == 'timeout'])
                    m_len = len(full_table)
                    summary_dict[column + "_perc_timeout"] = n_timeouts / m_len * 100

        # Re-sort by method, exploiting that the columns are ordered per method.
        latex_table_dict = {}
        for counter, key in enumerate(summary_dict):
            c_key = key.split("_")[0]
            if c_key in latex_table_dict:
                latex_table_dict[c_key].append(summary_dict[key])
            else:
                latex_table_dict[c_key] = [summary_dict[key]]

        latex_table = pd.DataFrame(latex_table_dict)
        latex_table = latex_table.rename(columns={"BSAT": "%Timeout", "BBran": "Sub-problems", "BTime": "Time [s]"})
        latex_tables.append(latex_table[latex_table.columns[::-1]])

    merged_plot_names = [cname.split(" ")[0] for cname in plot_names]
    merged_latex_table = pd.concat(latex_tables, axis=1, keys=merged_plot_names)
    print(merged_latex_table)
    converted_latex_table = merged_latex_table.to_latex(float_format="%.2f")
    print(f"latex table.\n Row names: {labels} \n Table: \n{converted_latex_table}")


def iclr_plots():

    # Plots for OVAL-CIFAR
    folder = './cifar_results/'
    timeout = 3600
    bases = ["base_100", "wide_100", "deep_100"]
    plot_names = ["Base model", "Wide large model", "Deep large model"]
    time_base = "BTime_KW"
    file_list_nobase = [
        "KW_prox_100-pinit-eta100.0-feta100.0.pkl",
        "KW_bigm-adam_180-pinit-ilr0.01,flr0.0001.pkl",
        "KW_cut_100_no_easy-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
        f"KW_cut_100-pinit-ilr0.001,flr1e-06-cut_add2.0-diilr0.01,diflr0.0001.pkl",
        "anderson-mip.pkl",
        f"KW_gurobi-anderson_1.pkl",
        "eran.pkl"
    ]
    time_name_list = [
        f"{time_base}_prox_100",
        f"{time_base}_bigm-adam_180",
        f"{time_base}_cut_100_no_easy",
        f"{time_base}_cut_100",
        f"BTime_anderson-mip",
        f"{time_base}_gurobi-anderson_1",
        f"{time_base}_eran"
    ]
    labels = [
        "BDD+ BaBSR",
        "Big-M BaBSR",
        "Active Set BaBSR",
        "Big-M + Active Set BaBSR",
        r"MIP $\mathcal{A}_k$",
        "G. Planet + G. 1 cut BaBSR",
        "ERAN"
    ]
    line_styles = [
        "dotted",
        "solid",
        "solid",
        "solid",
        "dotted",
        "dotted",
        "dotted",
    ]
    for base, plt_name in zip(bases, plot_names):
        file_list = [f"{base}_" + cfile for cfile in file_list_nobase]
        plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", plt_name,
                                create_csv=False, linestyles=line_styles)
    to_latex_table(folder, bases, file_list_nobase, time_name_list, labels, plot_names, timeout)
    plt.show()


if __name__ == "__main__":

    iclr_plots()
