# Legacy reference scripts

These files are the original scripts from the pre-refactor ZykloSim project.
They are kept for reference and traceability and are not part of the
maintained `cyclotron_sim` package.

| File | Purpose |
| --- | --- |
| `main.py` | Original PyQt/pyqtgraph front end. |
| `test2.py` | Early prototype of the GUI and plotting workflow. |
| `Draw_circle_nov24.py` | Small geometry/plotting experiment. |

The maintained application is `cyclotron_gui.py` in the repository root. It
reuses `ZykloSim_nov24.ui` and replaces the original pyqtgraph canvas with a
Matplotlib canvas, so pyqtgraph is not required.

The legacy scripts may expect the original working-directory layout and are
not covered by the automated tests.
