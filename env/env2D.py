import math
import gym
import matplotlib.pyplot as plt
from collections import deque
import random
import numpy as np


class Simple2DEnv(gym.Env):

    def __init__(
            self,
            world_size: float = 10.0,
            robot_size: float = 0.2,
            dt: float = 0.1,
            max_lin_vel: float = 0.25,
            max_ang_vel: float = np.pi,
            max_lin_acc: float = 0.5,
            max_ang_acc: float = np.pi,
            goal_radius: float = 0.3,
            seed: int | None = None,
            difficulty: str = 'train',
            use_lidar: bool = True,
            max_steps: int = 1000
    ):
        super().__init__()

        # Параметры среды симуляции
        self.world_size = world_size
        self.robot_size = robot_size
        self.dt = dt
        self.goal_radius = goal_radius
        self.max_steps=max_steps
        self.steps = 0
        self.max_dist = math.sqrt(2) * self.world_size

        self.alpha = 5.0  # коэффициент за приближение
        self.gamma = 10.0  # бонус за достижение цели
        self.delta = -10.0  # штраф за столкновение
        self.eps = -0.2 # коэффициент штрафа за данные лидар
        self.eps1=-0.05 # коэффициент штрафа за ускорение 1
        self.eps2=-0.05 # коэффициент штрафа за ускорение 2

        # Динамика
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.max_lin_acc = max_lin_acc
        self.max_ang_acc = max_ang_acc
        self.last_distance = None
        self.obstacle_density = {
            'train': 5,
            'test': 6
        }[difficulty]

        self.max_static_obs = self.obstacle_density
        self.max_dynamic_obs = self.max_static_obs-2
        self.static_obs_size = (0.3, 0.5)
        self.dynamic_obs_size = (0.2, 0.3)
        self.dynamic_speed_range = (-self.max_lin_vel/2, self.max_lin_vel/2)

        # Лидар
        self.use_lidar = use_lidar
        self.lidar_num_rays = 64 if use_lidar else 0
        self.lidar_max_range = 4
        self.lidar_fov = 2 * np.pi
        self.lidar_noise_std = 0.01

        self.stack_size = 3
        self.lidar_history = deque(maxlen=self.stack_size)

        # Действие
        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_lin_acc, -self.max_ang_acc], dtype=np.float32),
            high=np.array([self.max_lin_acc, self.max_ang_acc], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Наблюдение
        obs_low = np.concatenate(
            [
                np.array(
                    [
                        -self.world_size / 2,-self.world_size / 2,0.0,  # pos
                        -self.max_lin_vel,-self.max_lin_vel,0.0,  # lin_vel
                        -self.max_ang_vel,-self.max_ang_vel,-self.max_ang_vel,  # ang_vel
                        -np.pi,  # heading_to_goal
                        0.0,  # dist_to_goal
                        0 # step
                    ],
                    dtype=np.float32,
                ),
                np.zeros(self.lidar_num_rays, dtype=np.float32),  # lidar min = 0
            ]
        )
        obs_high = np.concatenate(
            [
                np.array(
                    [
                        self.world_size / 2,self.world_size / 2,0.0,
                        self.max_lin_vel,self.max_lin_vel,0.0,
                        self.max_ang_vel,self.max_ang_vel,self.max_ang_vel,
                        np.pi,np.sqrt(2) * (self.world_size / 2), self.max_steps

                    ],
                    dtype=np.float32,
                ),
                np.full(self.lidar_num_rays, 1, dtype=np.float32),
            ]
        )
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)

        # ГСЧ
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._reset_state()

    def _reset_state(self):
        area = self.world_size / 2.0 - 0.2
        self.goal = np.array([
            random.uniform(-area, area),
            random.uniform(-area, area), 0
        ], dtype=np.float32)

        dist = random.uniform(5, self.world_size )
        ang = random.uniform(-math.pi, math.pi)
        x_coord = self.goal[0] + dist * math.cos(ang)
        y_coord = self.goal[1] + dist * math.sin(ang)
        while (x_coord<-area or x_coord>area) or (y_coord<-area or y_coord>area):
            dist = random.uniform(5, self.world_size)
            ang = random.uniform(-math.pi, math.pi)
            x_coord = self.goal[0] + dist * math.cos(ang)
            y_coord = self.goal[1] + dist * math.sin(ang)

        self.pos = np.array([
            x_coord,
            y_coord, 0
        ], dtype=np.float32)
        self.yaw = self.np_random.uniform(-np.pi, np.pi)

        # Скорости
        self.lin_vel = np.zeros(3, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)

        # Статические препятствия
        self.static_obs: list[dict] = []
        for _ in range(self.max_static_obs):
            t = np.random.uniform(0.4, 0.6)
            base_point = self.pos + t * (self.goal - self.pos)
            side_noise = [random.uniform(-1.5, 1.5),random.uniform(-1.5, 1.5),0]
            obs_pos = base_point + side_noise
            obs_pos = np.clip(obs_pos, -area, area)
            size = self.np_random.uniform(*self.static_obs_size)
            if (np.linalg.norm(obs_pos - self.pos) > 0.4 and
                    np.linalg.norm(obs_pos - self.goal) > 0.4):
                self.static_obs.append({"pos": obs_pos, "size": size})

        # Динамические препятствия
        self.dynamic_obs: list[dict] = []
        for _ in range(self.max_dynamic_obs):
            size = self.np_random.uniform(*self.dynamic_obs_size)
            pos1 = [random.uniform(-area, area), random.uniform(-area, area), 0]
            while (self.inCircle(pos1[0], pos1[1], self.pos[0], self.pos[1], size / 2 + self.robot_size / 2 + 1) or self.inCircle(
                    pos1[0], pos1[1], self.goal[0], self.goal[1], size / 2 + self.goal_radius + 1)):
                pos1 = [random.uniform(-area, area), random.uniform(-area, area), 0]

            angle = self.np_random.uniform(-np.pi, np.pi)
            speed = self.np_random.uniform(*self.dynamic_speed_range)
            vel = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32) * speed
            self.dynamic_obs.append({"pos": pos1, "size": size, "vel": vel})

        self.steps = 0
        vec_to_goal = self.goal - self.pos
        dist_to_goal = np.linalg.norm(vec_to_goal[:2])
        self.last_distance = dist_to_goal
        self.done = False

    def _update_dynamic_obstacles(self):
        half = self.world_size / 2
        for obs in self.dynamic_obs:
            if self.np_random.uniform(0, 1) < 0.02:
                speed = np.linalg.norm(obs["vel"])
                new_angle = self.np_random.uniform(-np.pi, np.pi)
                obs["vel"] = np.array([np.cos(new_angle), np.sin(new_angle),0], dtype=np.float32) * speed
            new_pos = obs["pos"] + obs["vel"] * self.dt
            rad = obs["size"] / 2

            # Отскок от стен
            if new_pos[0] - rad < -half:
                new_pos[0] = -half + rad
                obs["vel"][0] *= -1
            if new_pos[0] + rad > half:
                new_pos[0] = half - rad
                obs["vel"][0] *= -1
            if new_pos[1] - rad < -half:
                new_pos[1] = -half + rad
                obs["vel"][1] *= -1
            if new_pos[1] + rad > half:
                new_pos[1] = half - rad
                obs["vel"][1] *= -1

            obs["pos"] = new_pos

    def _check_collision(self) -> bool:
        for obs in self.static_obs:
            if self._aabb_overlap(self.pos[:2], self.robot_size / 2,
                                  obs["pos"][:2], obs["size"] / 2):
                return True
        for obs in self.dynamic_obs:
            if self._aabb_overlap(self.pos[:2], self.robot_size / 2,
                                  obs["pos"][:2], obs["size"] / 2):
                return True
        return False

    @staticmethod
    def _aabb_overlap(c1, r1, c2, r2) -> bool:
        return (abs(c1[0] - c2[0]) <= (r1 + r2)) and (abs(c1[1] - c2[1]) <= (r1 + r2))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        self.lidar_history.clear()
        raw_obs = self._get_obs()
        robot_part = raw_obs[:-self.lidar_num_rays]
        lidar_part = raw_obs[-self.lidar_num_rays:]
        return self._get_stacked_obs(robot_part, lidar_part)

    def step(self, action):
        if self.done:
            raise RuntimeError("Ошибка: step() после завершения эпизода")

        lin_acc = action[0]*self.max_lin_acc
        ang_acc = action[1]*self.max_ang_acc
        self.ang_vel[2] = np.clip(self.ang_vel[2] + ang_acc * self.dt,
                                  -self.max_ang_vel, self.max_ang_vel)

        self.yaw = (self.yaw + self.ang_vel[2] * self.dt) % (2 * np.pi)
        direction = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0], dtype=np.float32)
        self.lin_vel[:2] += (lin_acc * direction[:2]) * self.dt
        speed = np.linalg.norm(self.lin_vel[:2])
        if speed > self.max_lin_vel:
            self.lin_vel[:2] = self.lin_vel[:2] / speed * self.max_lin_vel

        self.pos += self.lin_vel * self.dt

        self._update_dynamic_obstacles()
        self.steps += 1

        reward, self.done, info = self._compute_reward(action)
        raw_nxt_obs = self._get_obs()
        robot_part = raw_nxt_obs[:-self.lidar_num_rays]
        lidar_part = raw_nxt_obs[-self.lidar_num_rays:]
        stacked_nxt_obs = self._get_stacked_obs(robot_part, lidar_part)
        return stacked_nxt_obs, reward, self.done, False, info

    def _compute_reward(self,action):
        done = False
        reward = 0.0
        lin_acc = action[0] * self.max_lin_acc
        ang_acc = action[1] * self.max_ang_acc
        vec_to_goal = self.goal - self.pos
        dist_to_goal = np.linalg.norm(vec_to_goal[:2])
        diff=0
        # +- за приближение
        if self.last_distance is not None:
            diff=(self.last_distance - dist_to_goal)
            reward += self.alpha * diff
        self.last_distance = dist_to_goal

        # Штраф за опасное сближение
        minV=min(self._get_lidar_scan())
        danger_range=0.375
        if minV<danger_range:
            reward += self.eps * (((danger_range - minV)/danger_range)**2)

        reward += self.eps1 * (lin_acc ** 2) + self.eps2 * (ang_acc ** 2)

        # Награда за достижение цели
        if (dist_to_goal) < self.goal_radius:
            reward += self.gamma
            done = True
            info = {"reason": "reached_goal", "step":{self.steps}}
            return reward, done, info
        else:
            info={}

        # Штраф за столкновение/выезд
        half = self.world_size / 2
        out_of_bounds = (
                self.pos[0] < -half + self.robot_size / 2 or
                self.pos[0] > half - self.robot_size / 2 or
                self.pos[1] < -half + self.robot_size / 2 or
                self.pos[1] > half - self.robot_size / 2
        )
        if out_of_bounds:
            done = True
            reward += self.delta
            info = {"reason": "out_of_bounds", "step":{self.steps}}
            return reward, done, info

        if self._check_collision():
            done = True
            reward += self.delta
            info = {"reason": "collision", "step":{self.steps}}
            return reward, done, info

        # штраф за истечение времени
        if self.steps >= self.max_steps:
            reward += self.delta/4
            done = True
            info = {"reason": "max_steps", "step":{self.steps}}
        return reward, done,info

    def _get_stacked_obs(self, robot_data, lidar_data):
        if len(self.lidar_history) == 0:
            for _ in range(self.stack_size):
                self.lidar_history.append(lidar_data)
        else:
            self.lidar_history.append(lidar_data)
        stacked_lidar = np.concatenate(list(self.lidar_history))
        return np.concatenate([robot_data, stacked_lidar]).astype(np.float32)

    def _get_obs(self):
        vec_to_goal = self.goal - self.pos
        dist_to_goal = np.linalg.norm(vec_to_goal[:2])/self.max_dist
        heading_to_goal = self._angle_to_target(vec_to_goal[:2])

        base = np.concatenate(
            [
            (self.pos / self.world_size + 0.5),
             self.lin_vel,
             self.ang_vel,
             np.array([heading_to_goal, dist_to_goal / self.max_dist, self.steps / self.max_steps],
                      dtype=np.float32),
            ]
         )
        if self.use_lidar==True:
            lidar = self._get_lidar_scan()
        else:
            lidar=[]
        return np.concatenate([base, lidar]).astype(np.float32)

    def _angle_to_target(self, vec_xy: np.ndarray) -> float:
        forward = np.array([np.cos(self.yaw), np.sin(self.yaw)])
        if np.linalg.norm(vec_xy) < 1e-6:
            return 0.0
        target_dir = vec_xy / np.linalg.norm(vec_xy)
        angle = np.arctan2(target_dir[1], target_dir[0]) - np.arctan2(forward[1], forward[0])
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return float(angle)

    def _cast_ray(self, angle_rad: float) -> float:
        dir_vec = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=np.float32)
        origin = self.pos.copy()
        min_dist = self.lidar_max_range
        half = self.world_size / 2
        for axis in range(2):  # 0 → x, 1 → y
            if abs(dir_vec[axis]) < 1e-8:
                continue
            t_pos = (half - origin[axis]) / dir_vec[axis]
            if t_pos > 0:
                other = 1 - axis
                y_at_t = origin[other] + t_pos * dir_vec[other]
                if -half <= y_at_t <= half:
                    min_dist = min(min_dist, t_pos)
            t_neg = (-half - origin[axis]) / dir_vec[axis]
            if t_neg > 0:
                other = 1 - axis
                y_at_t = origin[other] + t_neg * dir_vec[other]
                if -half <= y_at_t <= half:
                    min_dist = min(min_dist, t_neg)

        def ray_aabb(origin, dir_vec, center, half_size):
            t_min = -np.inf
            t_max = np.inf
            for i in range(2):
                if abs(dir_vec[i]) < 1e-8:
                    if not (center[i] - half_size <= origin[i] <= center[i] + half_size):
                        return np.inf
                else:
                    t1 = (center[i] - half_size - origin[i]) / dir_vec[i]
                    t2 = (center[i] + half_size - origin[i]) / dir_vec[i]
                    t_near = min(t1, t2)
                    t_far = max(t1, t2)
                    t_min = max(t_min, t_near)
                    t_max = min(t_max, t_far)
                    if t_min > t_max:
                        return np.inf
            return t_min if t_min > 0 else np.inf

        # Статические
        for obs in self.static_obs:
            d = ray_aabb(origin, dir_vec, obs["pos"], obs["size"] / 2)
            if d < min_dist:
                min_dist = d

        # Динамические
        for obs in self.dynamic_obs:
            d = ray_aabb(origin, dir_vec, obs["pos"], obs["size"] / 2)
            if d < min_dist:
                min_dist = d

        return min(min_dist, self.lidar_max_range)

    def _get_lidar_scan(self) -> np.ndarray:
        start_angle = self.yaw - self.lidar_fov / 2.0
        angles = start_angle + np.arange(self.lidar_num_rays) * (self.lidar_fov / (self.lidar_num_rays - 1))
        scans = np.empty(self.lidar_num_rays, dtype=np.float32)

        for i, a in enumerate(angles):
            dist = self._cast_ray(a)
            if self.lidar_noise_std > 0:
                dist += self.np_random.normal(0.0, self.lidar_noise_std)
                dist = np.clip(dist, 0.0, self.lidar_max_range)
            scans[i] = dist/self.lidar_max_range
        return scans

    def inCircle(self, x, y, x0, y0, r):
        dist_sq = (x - x0) ** 2 + (y - y0) ** 2
        return dist_sq <= r ** 2

    def render(self):
        try:
            import matplotlib.pyplot as plt
            from matplotlib import patches
        except ImportError:
            raise RuntimeError("требуется matplotlib")

        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        ax = self._ax
        ax.clear()
        ax.set_aspect("equal")
        half = self.world_size / 2
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)

        # Робот
        robot_patch = patches.Rectangle(
            (self.pos[0] - self.robot_size / 2, self.pos[1] - self.robot_size / 2),
            self.robot_size, self.robot_size,
            angle=np.degrees(self.yaw),
            linewidth=2,edgecolor="blue",facecolor="lightblue",
        )
        ax.add_patch(robot_patch)

        # Цель
        goal_patch = patches.Circle(
            (self.goal[0], self.goal[1]),
            radius=self.goal_radius,
            edgecolor="green",
            facecolor="none",
            linestyle="--",
            linewidth=2,
        )
        ax.add_patch(goal_patch)

        # Статические препятствия
        for obs in self.static_obs:
            sq = patches.Rectangle(
                (obs["pos"][0] - obs["size"] / 2, obs["pos"][1] - obs["size"] / 2),
                obs["size"],obs["size"],
                linewidth=1,edgecolor="black",facecolor="gray",alpha=0.6,
            )
            ax.add_patch(sq)

        # Динамические препятствия
        for obs in self.dynamic_obs:
            sq = patches.Rectangle(
                (obs["pos"][0] - obs["size"] / 2, obs["pos"][1] - obs["size"] / 2),
                obs["size"],obs["size"],
                linewidth=1,edgecolor="red",facecolor="orange",alpha=0.6,
            )
            ax.add_patch(sq)

        # Лучи лидара
        if self.use_lidar==True:
            scan = self._get_lidar_scan()
            start_angle = self.yaw - self.lidar_fov / 2.0
            angles = start_angle + np.arange(self.lidar_num_rays) * (self.lidar_fov / (self.lidar_num_rays - 1))
            for a, d in zip(angles, scan):
                end = self.pos[:2] + d * np.array([np.cos(a), np.sin(a)])
                ax.plot([self.pos[0]- self.robot_size / 2, end[0]], [self.pos[1]- self.robot_size / 2, end[1]],
                        color="cyan", linewidth=0.8, alpha=0.5)

        # Вектор к цели (для ориентации)
        ax.arrow(
            self.pos[0]- self.robot_size / 2,
            self.pos[1]- self.robot_size / 2,
            self.goal[0] - self.pos[0],
            self.goal[1] - self.pos[1],
            head_width=0.05,head_length=0.1,
            fc="green",ec="green",linewidth=1,
        )

        ax.set_title(f"Шаг {self.steps} | До цели {np.linalg.norm(self.goal[:2]-self.pos[:2]):.2f}м")
        plt.pause(0.1)

    def close(self):
        if hasattr(self, "_fig"):
            plt.close(self._fig)