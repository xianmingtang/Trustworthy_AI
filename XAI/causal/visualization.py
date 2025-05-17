"""
    Handle the data visualization.
        Including the causal graph, table, data analysis, etc.
"""
from causallearn.utils.GraphUtils import GraphUtils
from pathlib import Path

def causal_graph(cg, algorithm: str | None = None):
    """
    Render and save a pydot causal graph coming from a causallearn PC/FCI/... run.

    1) Converts cg.G → pydot
    2) Styles the title, fonts, margins
    3) Writes to ../notebooks/{algorithm}.png relative to cwd

    Prints out the final absolute path.
    """
    # 1) produce the pydot graph
    pydot_graph = GraphUtils.to_pydot(cg.G)

    # 2) style
    title = f"{algorithm} Learned Structure"
    pydot_graph.set('label', title)
    pydot_graph.set('labelloc', 't')
    pydot_graph.set('fontsize', '18')
    pydot_graph.set('fontname', 'Helvetica')
    pydot_graph.set('margin', '0.2')

    # 3) figure out output directory (<cwd parent>/notebooks/)
    cwd = Path.cwd()
    out_dir = (cwd.parent / 'notebooks') if cwd.name.lower() != 'notebooks' else cwd
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) write png
    out_path = out_dir / f"{algorithm}.png"
    pydot_graph.write_png(str(out_path))

    print(f"{algorithm} causal diagram saved to {out_path.resolve()}")
