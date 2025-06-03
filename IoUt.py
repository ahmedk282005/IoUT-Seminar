import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.animation import FuncAnimation


def CreateMatrixAndVector(anchorPositions, auvPosition):
    A = []
    b = []

    for anchorPosition in anchorPositions:
        distance = math.dist(anchorPosition, auvPosition)

        x = anchorPosition[0]
        y = anchorPosition[1]
        noisy_distance = distance + np.random.normal(0, 1)
        noisy_projected_distance = math.sqrt(math.pow(noisy_distance, 2) -
                                             math.pow(anchorPosition[2] - auvPosition[2], 2))

        A.append([2 * x, 2 * y, 1])
        b.append([math.pow(x, 2) + math.pow(y, 2) - math.pow(noisy_projected_distance, 2)])

        # draw circle
        r = noisy_projected_distance
        theta = np.linspace(0, 2 * np.pi, 100)

        x = x + r * np.cos(theta)
        y = y + r * np.sin(theta)
        z = auvPosition[2] + np.zeros_like(theta)

        ax3D.plot(x, y, z, color='red', linewidth=0.5, zorder=0)

    return np.array(A), np.array(b)


def LeastSquares(A, b):
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return x


nAnchors = 3
nAuvs = 1

posAnchors = [[]] * nAnchors
posAnchors[0] = [30, 30, -1]
posAnchors[1] = [-30, 30, -1]
posAnchors[2] = [-30, -30, -1]
#posAnchors[3] = [30, -30, -1]

posAuvs = [[]] * nAuvs
posAuvs[0] = [15, 10, -25]

areaLimX = [-50, 50]
areaLimY = [-50, 50]
areaLimZ = [-50, 0]


fig = plt.figure(figsize=(8, 6))
ax3D = fig.add_subplot(121, projection='3d', computed_zorder=False)
axText = fig.add_subplot(122)

ax3D.set_xlim([areaLimX[0], areaLimX[1]])
ax3D.set_ylim([areaLimY[0], areaLimY[1]])
ax3D.set_zlim([areaLimZ[0], areaLimZ[1]])

ax3D.set_xlabel('X Position (m)')
ax3D.set_ylabel('Y Position (m)')
ax3D.set_zlabel('Z Position (m)')
ax3D.set_title('Localization Simulation')


ax3D.plot_surface(np.array([[areaLimX[0], areaLimX[1]], [areaLimX[0], areaLimX[1]]]),
                  np.array([[areaLimY[0], areaLimY[0]], [areaLimY[1], areaLimY[1]]]),
                  np.array([[areaLimZ[0], areaLimZ[0]], [areaLimZ[0], areaLimZ[0]]]),
                  color='brown', alpha=0.6, rstride=100, cstride=100, zorder=0)

for i, posAuv in enumerate(posAuvs):
    ax3D.scatter3D(posAuv[0], posAuv[1], posAuv[2], c='blue', s=8, zorder=1)
    ax3D.text3D(posAuv[0], posAuv[1], posAuv[2], f'AUV{i + 1}',
                fontsize=12, color='black', ha='left', va='bottom', zorder=1)
    axText.text(0.5, 1 - i * 0.1, f'AUV{i + 1} Position\nX: {posAuv[0]}\nY: {posAuv[1]}\nZ: {posAuv[2]}',
                fontsize=9, color='black', ha='left', va='top')

for i, posAnchor in enumerate(posAnchors):
    ax3D.scatter3D(posAnchor[0], posAnchor[1], posAnchor[2], color='red', zorder=1)
    ax3D.text3D(posAnchor[0], posAnchor[1], posAnchor[2], f'A{i + 1}',
                fontsize=12, color='black', ha='left', va='bottom', zorder=1)
    axText.text(0.3, 1 - i * 0.1, f'A{i + 1} Position:\nX: {posAnchor[0]}\nY: {posAnchor[1]}\nZ: {posAnchor[2]}',
                fontsize=9, color='black', ha='left', va='top')

ax3D.plot_surface(np.array([[areaLimX[0], areaLimX[1]], [areaLimX[0], areaLimX[1]]]),
                  np.array([[areaLimY[0], areaLimY[0]], [areaLimY[1], areaLimY[1]]]),
                  np.array([[areaLimZ[1], areaLimZ[1]], [areaLimZ[1], areaLimZ[1]]]),
                  color='aqua', alpha=0.2, rstride=100, cstride=100, zorder=2)

A_g, b_g = CreateMatrixAndVector(posAnchors, posAuvs[0])
x_g = LeastSquares(A_g, b_g)

ax3D.scatter3D(x_g[0], x_g[1], posAuv[2], c='red', s=8, zorder=1)
axText.text(0.5, 0, f'AUV{1} Multilaterated\nX: {float(x_g[0]):.3f}\nY: {float(x_g[1]):.3f}\nZ: {float(posAuv[2]):.3f}',
            fontsize=9, color='black', ha='left', va='top')


# Update function for animation
def update(frame):
    ax3D.clear()
    axText.clear()

    ax3D.set_xlim([areaLimX[0], areaLimX[1]])
    ax3D.set_ylim([areaLimY[0], areaLimY[1]])
    ax3D.set_zlim([areaLimZ[0], areaLimZ[1]])

    ax3D.set_xlabel('X Position (m)')
    ax3D.set_ylabel('Y Position (m)')
    ax3D.set_zlabel('Z Position (m)')
    ax3D.set_title('Localization Simulation')

    ax3D.plot_surface(np.array([[areaLimX[0], areaLimX[1]], [areaLimX[0], areaLimX[1]]]),
                      np.array([[areaLimY[0], areaLimY[0]], [areaLimY[1], areaLimY[1]]]),
                      np.array([[areaLimZ[0], areaLimZ[0]], [areaLimZ[0], areaLimZ[0]]]),
                      color='brown', alpha=0.6, rstride=100, cstride=100, zorder=0)

    for i, posAuv in enumerate(posAuvs):
        ax3D.scatter3D(posAuv[0], posAuv[1], posAuv[2], c='blue', s=8, zorder=1)
        ax3D.text3D(posAuv[0], posAuv[1], posAuv[2], f'AUV{i + 1}',
                    fontsize=12, color='black', ha='left', va='bottom', zorder=1)
        axText.text(0.5, 1 - i * 0.1, f'AUV{i + 1} Position\nX: {posAuv[0]}\nY: {posAuv[1]}\nZ: {posAuv[2]}',
                    fontsize=9, color='black', ha='left', va='top')

    for i, posAnchor in enumerate(posAnchors):
        ax3D.scatter3D(posAnchor[0], posAnchor[1], posAnchor[2], color='red', zorder=1)
        ax3D.text3D(posAnchor[0], posAnchor[1], posAnchor[2], f'A{i + 1}',
                    fontsize=12, color='black', ha='left', va='bottom', zorder=1)
        axText.text(0.3, 1 - i * 0.1, f'A{i + 1} Position:\nX: {posAnchor[0]}\nY: {posAnchor[1]}\nZ: {posAnchor[2]}',
                    fontsize=9, color='black', ha='left', va='top')

    ax3D.plot_surface(np.array([[areaLimX[0], areaLimX[1]], [areaLimX[0], areaLimX[1]]]),
                      np.array([[areaLimY[0], areaLimY[0]], [areaLimY[1], areaLimY[1]]]),
                      np.array([[areaLimZ[1], areaLimZ[1]], [areaLimZ[1], areaLimZ[1]]]),
                      color='aqua', alpha=0.2, rstride=100, cstride=100, zorder=2)

    A_g, b_g = CreateMatrixAndVector(posAnchors, posAuvs[0])
    x_g = LeastSquares(A_g, b_g)

    ax3D.scatter3D(x_g[0], x_g[1], posAuvs[0][2], c='red', s=8, zorder=1)

    # Clear the text axis and add the new text
    axText.axis('off')
    axText.text(0.5, 0, f'AUV{1} Multilaterated\nX: {float(x_g[0]):.3f}\nY: {float(x_g[1]):.3f}\nZ: {float(posAuvs[0][2]):.3f}',
                fontsize=9, color='black', ha='left', va='top')

    return ax3D,

# Create the animation
ani = FuncAnimation(fig, update, frames=10, interval=1000, repeat=True)


areaLengthX = areaLimX[1] - areaLimX[0]
areaLengthY = areaLimY[1] - areaLimY[0]
areaLengthZ = areaLimZ[1] - areaLimZ[0]
ax3D.get_proj = lambda: np.dot(Axes3D.get_proj(ax3D),
                               np.diag([areaLengthX, areaLengthY, areaLengthZ,
                                        max(areaLengthX, areaLengthY, areaLengthZ)]))

axText.axis('off')
plt.tight_layout()
plt.show()