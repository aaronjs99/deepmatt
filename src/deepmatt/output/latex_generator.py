# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * University of California, Los Angeles                   *
# *                                                         *
# * Author : Aaron John-Sabu                                *
# * Contact: aaronjs@g.ucla.edu                             *
# *                                                         *
# ***********************************************************/

import numpy as np


def generate_latex_table(data, headers, caption, label, format_spec=None):
    """
    Generate a LaTeX table from data.

    Args:
        data: List of lists (rows)
        headers: List of column headers
        caption: Table caption
        label: Table label for referencing
        format_spec: Optional list of format strings for each column
    """
    n_cols = len(headers)

    if format_spec is None:
        format_spec = [".2f"] * n_cols

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"    \centering")
    latex.append(f"    \\caption{{{caption}}}")
    latex.append(f"    \\label{{tab:{label}}}")
    latex.append(f'    \\begin{{tabular}}{{{"l" + "r" * (n_cols - 1)}}}')
    latex.append(r"        \toprule")
    latex.append(f'        {" & ".join(headers)} \\\\')
    latex.append(r"        \midrule")

    for row in data:
        formatted_row = []
        for i, val in enumerate(row):
            if isinstance(val, str):
                formatted_row.append(val)
            elif isinstance(val, (int, float)):
                formatted_row.append(f"{val:{format_spec[i]}}")
            else:
                formatted_row.append(str(val))
        latex.append(f'        {" & ".join(formatted_row)} \\\\')

    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def save_results_table(results_dict, filename):
    """Save results as LaTeX table."""
    with open(filename, "w") as f:
        for table_name, table_data in results_dict.items():
            f.write(table_data + "\n\n")


# Example usage
if __name__ == "__main__":
    # Formation costs table
    data = [
        ["Deputy 1", "+10 km X", "0.86", "~24"],
        ["Deputy 2", "-10 km X", "0.86", "~24"],
        ["Officer 1", "+10 km Z", "0.41", "~11"],
        ["Officer 2", "-10 km Z", "0.41", "~11"],
    ]
    headers = ["Agent", "Position", "Total $\\Delta V$ (m/s)", "Annual Cost (m/s)"]

    table = generate_latex_table(
        data,
        headers,
        "Multi-agent formation Delta-V costs over 13 days",
        "formation_costs",
    )

    print(table)
