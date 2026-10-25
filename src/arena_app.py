"""The live EKF arena app (provided).

Drives the robot through the museum while two EKF localizers -- one
corrected with the beam model, one with the likelihood field, both moved
by your velocity motion model -- track it in real time.  The panel shows
one of them at a time; press 1 or 2 to toggle between the sensor models.
Requires your task 1 sampler and your task 4 EKF (see src/ekf.py,
Table 7.2).

Controls
--------
1 / 2           beam model / likelihood field      A   toggle autopilot
UP/DOWN/LEFT/RIGHT  drive manually                K   kidnap the robot
B / S / T       update terms / samples / trails   R   reset both filters
SPACE           pause                             ESC quit
"""

from __future__ import annotations

import numpy as np
import pygame

from . import config
from .ekf import EKFLocalizer
from .motion_models import sample_velocity_motion_model
from .robot import Robot
from .tour import TourDriver
from .utils import make_rng
from .world import World

BEAM_COLOR = (45, 212, 191)       # teal
FIELD_COLOR = (244, 114, 182)     # pink
SAMPLE_COLOR = (125, 211, 252)    # light blue
N_SAMPLES = 80


class ArenaApp:
    def __init__(self, seed=10, speed=1.0, autopilot=True,
                 start_pose=(5.6, 8.0, np.pi / 2)):
        self.world = World.museum()
        self.robot = Robot(self.world, start_pose=start_pose, seed=seed)
        self.tour = TourDriver(self.world)
        self.rng = make_rng(seed + 3)

        # the two parallel localizers, both started near the true pose
        # (EKF localization assumes a Gaussian initial belief)
        self.ekf_beam = EKFLocalizer(self.world, sensor="beam",
                                     init_pose=self.robot.pose, seed=seed + 1)
        self.ekf_field = EKFLocalizer(self.world, sensor="likelihood",
                                      init_pose=self.robot.pose, seed=seed + 2)

        self.speed = speed
        self.autopilot = autopilot
        self.active_sensor = "beam"   # 1 = beam model, 2 = likelihood field
        self.show_beams = True
        self.show_samples = True
        self.show_trails = True
        self.trail_true = [self.robot.pose[:2].copy()]
        self.trail_beam = [self.ekf_beam.mu[:2].copy()]
        self.trail_field = [self.ekf_field.mu[:2].copy()]
        self.samples = np.empty((0, 3))
        self.last_cmd = (0.0, 0.0)
        self.sim_time = 0.0
        self.paused = False

        self.scale = 54
        self.margin = 14
        self.hud_h = 96
        w_px = int(self.world.width * self.scale) + 2 * self.margin
        h_px = int(self.world.height * self.scale) + 2 * self.margin + self.hud_h
        pygame.init()
        pygame.display.set_caption("EKF Arena -- probabilistic robotics, chapters 5-7")
        self.screen = pygame.display.set_mode((w_px, h_px))
        # the bundled default font loads on every platform without X11 lookups
        self.font = pygame.font.Font(None, 18)
        self.clock = pygame.time.Clock()
        self.map_surface = self._render_map()

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def to_px(self, x, y):
        u = self.margin + x * self.scale
        v = self.margin + (self.world.height - y) * self.scale
        return u, v

    def _render_map(self):
        s = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        s.fill("#14171c")
        cell = max(1, int(np.ceil(self.scale * self.world.res)))
        for iy, ix in np.argwhere(self.world.grid):
            u, v = self.to_px(ix * self.world.res, (iy + 1) * self.world.res)
            s.fill("#596069", (u, v, cell, cell))
        return s

    def _draw_pose(self, pose, color, radius, width=0):
        u, v = self.to_px(pose[0], pose[1])
        r = radius * self.scale
        if width == 0:
            pygame.draw.circle(self.screen, color, (u, v), int(r))
        else:
            pygame.draw.circle(self.screen, color, (u, v), int(r), width)
        tip = (u + (r + 4) * np.cos(-pose[2]), v + (r + 4) * np.sin(-pose[2]))
        pygame.draw.line(self.screen, color, (u, v), tip, 3)

    def _draw_ellipse(self, mu, Sigma, color):
        """3-sigma covariance ellipse of the position estimate -- large
        enough that the beam vs likelihood-field difference shows."""
        C = Sigma[:2, :2]
        vals, vecs = np.linalg.eigh(C)
        a, b = 3.0 * np.sqrt(np.maximum(vals, 0.0)) * self.scale
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        u, v = self.to_px(mu[0], mu[1])
        if a < 1 or b < 1:
            return
        pts = [(u + a * np.cos(t) * np.cos(np.radians(ang)) - b * np.sin(t) * np.sin(np.radians(ang)),
                v + a * np.cos(t) * np.sin(np.radians(ang)) + b * np.sin(t) * np.cos(np.radians(ang)))
               for t in np.linspace(0, 2 * np.pi, 24)]
        pygame.draw.polygon(self.screen, color, pts, 2)

    def active(self):
        """The localizer currently shown in the panel."""
        return self.ekf_beam if self.active_sensor == "beam" else self.ekf_field

    def active_color(self):
        return BEAM_COLOR if self.active_sensor == "beam" else FIELD_COLOR

    def active_trail(self):
        return self.trail_beam if self.active_sensor == "beam" else self.trail_field

    def draw(self):
        self.screen.blit(self.map_surface, (0, 0))

        if self.show_trails:
            trails = [(self.trail_true, (138, 109, 59)), (self.active_trail(), self.active_color())]
            for trail, col in trails:
                if len(trail) > 1:
                    pts = [self.to_px(p[0], p[1]) for p in trail]
                    pygame.draw.aalines(self.screen, col, False, pts)

        if self.show_samples and len(self.samples):
            for p in self.samples:
                u, v = self.to_px(p[0], p[1])
                pygame.draw.circle(self.screen, SAMPLE_COLOR, (u, v), 2)

        if self.show_beams:
            self._draw_update_terms()

        active = self.active()
        self._draw_ellipse(active.mu, active.Sigma, self.active_color())
        self._draw_pose(active.mu, self.active_color(), 0.1, width=2)
        self._draw_pose(self.robot.pose, (249, 115, 22), self.robot.radius)
        self._draw_hud()

    def _draw_update_terms(self):
        """Visualize what the shown sensor update actually computes.

        The beam model ray-casts a sparse beam set from the estimate
        (lines, the expensive physics); the likelihood field only looks
        up the endpoints of (nearly) every beam in the pre-computed
        table (dots, cheap).
        """
        active = self.active()
        info = active.last_update_info
        if info is None:
            return
        mu = active.mu
        th = mu[2] + info["angles"]
        ex = mu[0] + info["z"] * np.cos(th)
        ey = mu[1] + info["z"] * np.sin(th)
        u0, v0 = self.to_px(mu[0], mu[1])
        col = (255, 235, 150)
        if info["kind"] == "rays":
            for x, y in zip(ex, ey):
                u1, v1 = self.to_px(x, y)
                pygame.draw.line(self.screen, col, (u0, v0), (u1, v1), 1)
        for x, y in zip(ex, ey):
            u1, v1 = self.to_px(x, y)
            pygame.draw.circle(self.screen, col, (u1, v1), 2)

    def _draw_hud(self):
        hud = pygame.Rect(0, self.screen.get_height() - self.hud_h,
                          self.screen.get_width(), self.hud_h)
        pygame.draw.rect(self.screen, "#0b0d10", hud)
        pygame.draw.line(self.screen, "#2a2f37", hud.topleft, hud.topright, 1)
        err_pos, err_th = self.active().error(self.robot.pose)
        active = self.active()
        n_used = 0 if active.last_update_info is None else len(active.last_update_info["z"])
        lines = [
            f"t = {self.sim_time:6.1f} s   showing: {self.active_sensor} EKF   "
            f"autopilot [{'on' if self.autopilot else 'off'}]",
            f"error {err_pos:5.2f} m / {np.degrees(err_th):5.1f} deg   "
            f"update {active.mean_update_ms:5.2f} ms ({n_used} beams)",
            "1 beam  2 likelihood field  A autopilot  K kidnap  R reset  "
            "B update terms  S samples  T trails  SPACE pause  ESC quit",
        ]
        for i, txt in enumerate(lines):
            self.screen.blit(self.font.render(txt, True, "#c9d1d9"),
                             (self.margin, hud.y + 6 + i * 17))

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------

    def control(self, keys):
        if self.autopilot:
            return self.tour.control(self.robot)
        v_cmd = w_cmd = 0.0
        if keys[pygame.K_UP]:
            v_cmd += 0.9
        if keys[pygame.K_DOWN]:
            v_cmd -= 0.6
        if keys[pygame.K_LEFT]:
            w_cmd += 1.2
        if keys[pygame.K_RIGHT]:
            w_cmd -= 1.2
        return v_cmd, w_cmd

    def _reset_filters(self):
        self.ekf_beam.__init__(self.world, sensor="beam",
                               init_pose=self.robot.pose, seed=1)
        self.ekf_field.__init__(self.world, sensor="likelihood",
                                init_pose=self.robot.pose, seed=2)

    def _clear_trails(self):
        """Drop every trail, ground truth included, and restart them from
        the current poses -- toggling the sensor model or resetting must
        redraw the trajectories instead of overwriting each other."""
        self.trail_true = [self.robot.pose[:2].copy()]
        self.trail_beam = [self.ekf_beam.mu[:2].copy()]
        self.trail_field = [self.ekf_field.mu[:2].copy()]

    def step_once(self, v_cmd, w_cmd):
        self.last_cmd = (v_cmd, w_cmd)
        self.robot.step(v_cmd, w_cmd)
        self.ekf_beam.predict((v_cmd, w_cmd))
        self.ekf_field.predict((v_cmd, w_cmd))
        if int(round(self.sim_time / config.DT)) % config.EKF.update_every == 0:
            scan = self.robot.scan()
            self.ekf_beam.update(scan, self.robot.laser_angles)
            self.ekf_field.update(scan, self.robot.laser_angles)
            # visualize the motion uncertainty feeding the shown filter:
            # samples of the velocity model around its predicted mean
            self.samples = sample_velocity_motion_model(
                self.last_cmd,
                np.tile(self.active().mu, (N_SAMPLES, 1)),
                config.VELOCITY_ALPHAS, config.DT, self.rng,
            )
        self.trail_true.append(self.robot.pose[:2].copy())
        self.trail_beam.append(self.ekf_beam.mu[:2].copy())
        self.trail_field.append(self.ekf_field.mu[:2].copy())
        self.sim_time += config.DT

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self, max_steps=None):
        steps = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    k = event.key
                    if k in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        return
                    elif k == pygame.K_1:
                        self.active_sensor = "beam"
                        self._clear_trails()
                    elif k == pygame.K_2:
                        self.active_sensor = "likelihood"
                        self._clear_trails()
                    elif k == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif k == pygame.K_a:
                        self.autopilot = not self.autopilot
                    elif k == pygame.K_b:
                        self.show_beams = not self.show_beams
                    elif k == pygame.K_s:
                        self.show_samples = not self.show_samples
                    elif k == pygame.K_t:
                        self.show_trails = not self.show_trails
                    elif k == pygame.K_r:
                        self._reset_filters()
                        self._clear_trails()
                    elif k == pygame.K_k:
                        # kidnap the robot; a Gaussian filter cannot
                        # re-localize globally, so the trackers restart
                        # from the new pose
                        pose = self.world.sample_free_poses(1, make_rng(steps))[0]
                        self.robot.pose = pose.copy()
                        self._reset_filters()
                        self._clear_trails()

            if not self.paused:
                v_cmd, w_cmd = self.control(pygame.key.get_pressed())
                self.step_once(v_cmd, w_cmd)
                steps += 1
                if max_steps and steps >= max_steps:
                    pygame.quit()
                    return

            self.draw()
            pygame.display.flip()
            self.clock.tick(30 * self.speed)
