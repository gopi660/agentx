"""Faux-3D ASCII mathematical rotating wireframe telemetry widget.

Renders an understated rotating geometric wireframe object projected to 2D
using 3D Euler rotation matrices and perspective division.
Disabled by default; runs on a non-blocking timer only when explicitly toggled.
"""

import math
from typing import List, Tuple
from rich.text import Text
from textual.widgets import Static


class Wireframe3DWidget(Static):
    """Faux-3D vector telemetry animation widget (disabled by default)."""

    def __init__(self, enabled: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.enabled = enabled
        self.angle_x: float = 25.0
        self.angle_y: float = 35.0
        self.angle_z: float = 10.0
        self._timer = None

        # 3D vertices for dual-box structure: Outer Box + Inner Gyro core
        self.outer_vertices: List[List[float]] = [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ]
        self.outer_edges: List[Tuple[int, int]] = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        self.core_vertices: List[List[float]] = [
            [0.0, -0.6, 0.0], [0.0, 0.6, 0.0],
            [-0.6, 0.0, 0.0], [0.6, 0.0, 0.0],
            [0.0, 0.0, -0.6], [0.0, 0.0, 0.6],
        ]
        self.core_edges: List[Tuple[int, int]] = [
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 4), (4, 3), (3, 5), (5, 2),
        ]

    def on_mount(self) -> None:
        """Initialize frame state on mount."""
        self.display = self.enabled
        self.update_frame()
        if self.enabled:
            self._timer = self.set_interval(0.1, self.step)

    def step(self) -> None:
        """Advance rotation angles and update the wireframe canvas."""
        if not self.enabled:
            return
        self.angle_x = (self.angle_x + 3.0) % 360.0
        self.angle_y = (self.angle_y + 4.5) % 360.0
        self.angle_z = (self.angle_z + 1.5) % 360.0
        self.update_frame()

    def toggle_animation(self) -> bool:
        """Toggle animation between active and disabled states."""
        self.enabled = not self.enabled
        self.display = self.enabled
        if self.enabled:
            if not self._timer:
                self._timer = self.set_interval(0.1, self.step)
            self.update_frame()
        else:
            if self._timer:
                self._timer.stop()
                self._timer = None
        return self.enabled

    def stop_animation(self) -> None:
        """Cleanly stop animation on widget unmount or shutdown."""
        if self._timer:
            self._timer.stop()
            self._timer = None
        self.enabled = False
        self.display = False

    def update_frame(self) -> None:
        """Render current 3D state to an understated ASCII canvas."""
        if not self.enabled:
            return

        width = 28
        height = 7

        rad_x = math.radians(self.angle_x)
        rad_y = math.radians(self.angle_y)
        rad_z = math.radians(self.angle_z)

        cx, sx = math.cos(rad_x), math.sin(rad_x)
        cy, sy = math.cos(rad_y), math.sin(rad_y)
        cz, sz = math.cos(rad_z), math.sin(rad_z)

        def project_3d_point(pt: List[float], scale_factor: float = 9.0) -> Tuple[int, int]:
            x, y, z = pt
            x1 = x * cy + z * sy
            z1 = -x * sy + z * cy
            y2 = y * cx - z1 * sx
            z2 = y * sx + z1 * cx
            x3 = x1 * cz - y2 * sz
            y3 = x1 * sz + y2 * cz

            dist = 3.2
            inv_z = 1.0 / (z2 + dist)
            px = int(width / 2 + (x3 * scale_factor) * inv_z)
            py = int(height / 2 + (y3 * scale_factor * 0.45) * inv_z)
            return px, py

        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Project outer box
        proj_outer = [project_3d_point(v, 9.5) for v in self.outer_vertices]
        for p1_i, p2_i in self.outer_edges:
            x1, y1 = proj_outer[p1_i]
            x2, y2 = proj_outer[p2_i]
            steps = max(abs(x2 - x1), abs(y2 - y1), 1)
            for s in range(steps + 1):
                px = int(x1 + (x2 - x1) * s / steps)
                py = int(y1 + (y2 - y1) * s / steps)
                if 0 <= px < width and 0 <= py < height:
                    grid[py][px] = "."

        for px, py in proj_outer:
            if 0 <= px < width and 0 <= py < height:
                grid[py][px] = "O"

        # Project inner core
        proj_core = [project_3d_point(v, 6.5) for v in self.core_vertices]
        for p1_i, p2_i in self.core_edges:
            x1, y1 = proj_core[p1_i]
            x2, y2 = proj_core[p2_i]
            steps = max(abs(x2 - x1), abs(y2 - y1), 1)
            for s in range(steps + 1):
                px = int(x1 + (x2 - x1) * s / steps)
                py = int(y1 + (y2 - y1) * s / steps)
                if 0 <= px < width and 0 <= py < height:
                    if grid[py][px] == " ":
                        grid[py][px] = "+"

        for px, py in proj_core:
            if 0 <= px < width and 0 <= py < height:
                grid[py][px] = "*"

        raw_str = "\n".join("".join(row) for row in grid)

        text = Text()
        text.append("Telemetry Wireframe (Ctrl+T)\n", style="#71717a")
        for line in raw_str.splitlines():
            text.append(line + "\n", style="#5ea28e")

        self.update(text)
