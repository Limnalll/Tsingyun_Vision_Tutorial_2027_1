"""The guided tour (provided): waypoints and a pursuit controller.

The waypoints are laid out so that every leg of the tour keeps a
comfortable clearance from obstacles (through door centers and between
pillars).  On top of that, the controller

* slows down and steers around whatever blocks the path ahead, and
* backs out of stalls, so the robot can never wedge itself against a
  pillar and give up.
"""

from __future__ import annotations

import numpy as np

from .planner import clear_line, plan_path
from .utils import wrap_angle

TOUR = [
    (8.9, 7.0),    # the hub: east of the exhibition block, chicane gap
    (8.0, 7.5),    # dip below the north-west pillar
    (8.0, 9.0),    # thread between the north pillar pair
    (9.0, 10.4),   # swing north around the north-east pillar
    (11.1, 9.1),   # straight through the east doorway (north)
    (12.6, 9.4),   # north-east gallery
    (12.8, 6.0),   # east wing, through the divider door
    (14.5, 1.9),   # south-east room
    (11.1, 2.3),   # straight through the east doorway (south)
    (10.1, 2.2),   # weave east of the south-east pillar
    (8.0, 2.2),    # pass under the gap between the south pillars
    (6.6, 2.2),    # weave west of the south-west pillar
    (4.9, 2.9),    # straight through the west doorway (south)
    (1.8, 1.8),    # south-west room
    (2.6, 3.0),
    (2.6, 7.2),    # up the west wing, through the divider door
    (1.3, 10.0),   # swing west around the twin pillar
    (2.6, 10.4),   # north-west gallery
    (4.9, 9.1),    # straight through the west doorway (north), back to the hub
]

REACH = 0.45       # waypoint reached within this radius
STALL_STEPS = 12   # barely-moving steps before backing out
EVADE_STEPS = 30   # how long to back out
FRONT_RANGE = 1.5  # look-ahead distance for obstacle steering


class TourDriver:
    """Steers a robot along ``TOUR`` forever.

    Rather than chasing the tour waypoint in a straight line (which
    wedges the robot into obstacles after a kidnap), the driver plans an
    A* path to it and follows that, replanning whenever the robot drifts
    off the path -- teleporting it somewhere counts as drifting.
    """

    def __init__(self, world, tour=TOUR, reach=REACH):
        self.world = world
        self.tour = list(tour)
        self.reach = reach
        self.wp = 0
        self.path = []
        self.state = {"evade": 0}

    def control(self, robot):
        target = self.tour[self.wp]
        if np.hypot(target[0] - robot.pose[0], target[1] - robot.pose[1]) < self.reach:
            self.wp = (self.wp + 1) % len(self.tour)
            self.path = []
            target = self.tour[self.wp]

        x, y = robot.pose[0], robot.pose[1]
        # replan when there is no path, the robot left it (kidnap!), or a
        # stall recovery just fired
        if self.state.pop("replan", False):
            self.path = []
        off_path = (
            not self.path
            or np.hypot(x - self.path[0][0], y - self.path[0][1]) > 1.0
            or not clear_line(self.world, (x, y), self.path[0])
        )
        if off_path:
            self.path = plan_path(self.world, (x, y), target) or [target]

        # consume intermediate path points as they are reached
        while len(self.path) > 1 and np.hypot(x - self.path[0][0], y - self.path[0][1]) < 0.35:
            self.path.pop(0)
        subgoal = self.path[0] if len(self.path) > 1 else target
        return pursuit(robot, subgoal, self.world, self.state)


def pursuit(robot, target, world, state):
    """Velocity command steering ``robot`` toward ``target``.

    ``state`` is a dict carried between calls (stall bookkeeping); the
    returned command is ``(v, omega)``.
    """
    x, y, th = robot.pose
    state.setdefault("evade", 0)

    # ---- stall recovery: back away while turning toward open space ----
    if state["evade"] > 0:
        state["evade"] -= 1
        return -0.5, state.get("evade_dir", 1.2)

    des = np.arctan2(target[1] - y, target[0] - x)
    err = wrap_angle(des - th)
    w_cmd = float(np.clip(2.2 * err, -1.6, 1.6))
    v_cmd = 0.9 * max(0.0, np.cos(err)) ** 2

    # ---- preventive steering: bend away from whatever blocks ahead -----
    front = world.raycast([[x, y]], [th], FRONT_RANGE)[0, 0]
    if front < 1.2:
        left = world.raycast([[x, y]], [th + 0.6], FRONT_RANGE)[0, 0]
        right = world.raycast([[x, y]], [th - 0.6], FRONT_RANGE)[0, 0]
        side = 1.0 if left > right else -1.0
        w_cmd = float(np.clip(w_cmd + side * 1.2 * (1.2 - front) / 1.2, -1.8, 1.8))
        v_cmd *= max(0.15, front / 1.2)

    # ---- stall detection: pressing into an obstacle without moving -----
    moved = np.hypot(x - state.get("lx", x), y - state.get("ly", y))
    if v_cmd > 0.2 and moved < 0.003:
        state["stall"] = state.get("stall", 0) + 1
        if state["stall"] > STALL_STEPS:
            left = world.raycast([[x, y]], [th + 1.2], 2.0)[0, 0]
            right = world.raycast([[x, y]], [th - 1.2], 2.0)[0, 0]
            state["evade_dir"] = 1.2 if left > right else -1.2
            state["evade"] = EVADE_STEPS
            state["stall"] = 0
            state["replan"] = True   # ask the driver for a fresh path
    else:
        state["stall"] = 0
    state["lx"], state["ly"] = x, y
    return v_cmd, w_cmd
