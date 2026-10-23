import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# pyqtgraph alias pg stellt die Graphikelemente wie z.B. mkPen(..) zur Verfügung
import pyqtgraph as pg

# PyQt6 stellt insbesondere QtWidgets zur Verfügung, welches als Parameter den Zeichenfkt übergeben wird
from PyQt6 import QtCore


def drawcircle(graphwidget, r):
    t = np.linspace(0, 2*math.pi, 100)
    x = np.cos(t)*r
    y = np.sin(t)*r
    pen = pg.mkPen(color=(0, 255, 0),
                   style=QtCore.Qt.PenStyle.SolidLine, width=4)

    graphwidget.plot(x, y, pen=pen)

    return graphwidget


def kammerplot(graphwidget, d, r):
    # Zeichnen der Kammer
    t = np.linspace(0, 2*math.pi, 100)
    x = np.cos(t)*r*1.02
    y = np.sin(t)*r*1.02
    pen = pg.mkPen(color=(0, 0, 0),
                   style=QtCore.Qt.PenStyle.DashDotDotLine, width=4)
    graphwidget.plot(
        x, y, pen=pen)
    graphwidget.setBackground("w")
    graphwidget.showGrid(x=True, y=True)

    # Zeichnen des oberen Randes des Dees
    alpha = np.arcsin(d/(2*r))
    t = np.linspace(alpha, math.pi-alpha, 100)
    x = np.cos(t)*0.7
    y = np.sin(t)*0.7
    pen = pg.mkPen(color=(0, 0, 0),
                   style=QtCore.Qt.PenStyle.SolidLine, width=4)
    graphwidget.plot(
        x, y, pen=pen)

    # Zeichnen der unteren Kante des Dees
    x = np.linspace(-r*np.cos(alpha), r*np.cos(alpha), 100)
    y = np.linspace(0.1, 0.1, 100)
    graphwidget.plot(
        x, y, pen=pen)

    # Zeichnen der oberen Kante des Dummy-Dees
    x = np.linspace(-r*np.cos(alpha), r*np.cos(alpha), 100)
    y = np.linspace(-0.1, -0.1, 100)
    graphwidget.plot(
        x, y, pen=pen)

    # Zeichnen des Dummy-Dees
    phi = np.arcsin(3*d/(2*r))
    t = np.linspace(math.pi+alpha, math.pi + phi, 100)
    x = np.cos(t)*r
    y = np.sin(t)*r
    graphwidget.plot(x, y, pen=pen)
    graphwidget.plot(-x, y, pen=pen)

    # Zeichnen der unteren Kante d. Dummy-Dees
    x = np.linspace(-np.cos(math.pi+phi)*0.7,
                    np.cos(math.pi+phi)*0.7, 100)
    y = np.linspace(-0.3, -0.3, 100)
    graphwidget.plot(x, y, pen=pen)
    return graphwidget


if __name__ == "__main__":
    t = np.linspace(0, 2*math.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    # Create just a figure and only one subplot
    fig, ax = plt.subplots()
# ax.plot(x, y)
    ax.set_title('Simple plot')

    animated_plot, = ax.plot([], [])
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('equal')

    def update_data(frame):
        animated_plot.set_data(x[:frame], y[:frame])
        return animated_plot

    animation = FuncAnimation(
        fig=fig,
        func=update_data,
        frames=len(x),
        interval=0.5,
        repeat=False,
    )

    plt.show()
