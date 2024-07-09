import torch


def torch_rand_float(low, high, size, device=None):
    return (high - low) * torch.rand(size, device=device) + low


class ActuatorFeature:
    def __init__(self,
                 num_envs,
                 num_joints_use,
                 device=None,
                 a: float = 1.0,
                 b: float = 1.0):
        self.num_envs = num_envs
        self.num_joints_use = num_joints_use
        if device is None:
            self.device = "cuda:0"
        else:
            self.device = device

        # delay feature
        self.a = a
        self.b = b
        self.alpha = torch.exp(torch.tensor([-1 / self.b]).to("cuda:0"))
        self.beta = self.a / self.b
        self.torques_delayed = torch.zeros(self.num_envs, self.num_joints_use, dtype=torch.float, device=self.device)

        # friction feature
        self.friction_param_0 = torch.zeros(self.num_envs, self.num_joints_use, dtype=torch.float, device=self.device)
        self.friction_param_1 = torch.zeros(self.num_envs, self.num_joints_use, dtype=torch.float, device=self.device)
        self.friction_param_2 = torch.zeros(self.num_envs, self.num_joints_use, dtype=torch.float, device=self.device)
        self.friction_param_3 = torch.zeros(self.num_envs, self.num_joints_use, dtype=torch.float, device=self.device)
        self.friction_param_4 = torch.zeros(self.num_envs, self.num_joints_use, dtype=torch.float, device=self.device)
        self.torques_friction = torch.zeros(self.num_envs, self.num_joints_use, dtype=torch.float, device=self.device)

    def reset(self, env_ids):
        self.torques_delayed[env_ids] = 0

        self.friction_param_0[env_ids] = torch.zeros(len(env_ids), self.num_joints_use, dtype=torch.float, device=self.device)
        self.friction_param_1[env_ids] = torch.zeros(len(env_ids), self.num_joints_use, dtype=torch.float, device=self.device)
        self.friction_param_2[env_ids] = torch.zeros(len(env_ids), self.num_joints_use, dtype=torch.float, device=self.device)
        self.friction_param_3[env_ids] = torch.zeros(len(env_ids), self.num_joints_use, dtype=torch.float, device=self.device)
        self.friction_param_4[env_ids] = torch.zeros(len(env_ids), self.num_joints_use, dtype=torch.float, device=self.device)

    def delay(self, torques):
        if torques.dim() == 1:
            torques = torques.unsqueeze(1)

        self.torques_delayed = self.alpha * self.torques_delayed + self.beta * torques

        return self.torques_delayed.clone()

    def friction(self, torques, velocities):
        flag_0 = (velocities <= 0.002) & (velocities >= -0.002)
        flag_1 = ((velocities > 0.002) & (velocities <= 0.16))
        flag_2 = (velocities > 0.16)
        flag_3 = ((velocities < -0.002) & (velocities >= -0.16))
        flag_4 = (velocities < -0.16)

        torques_friction = self.friction_param_0[:] / 0.002 * velocities[:] * flag_0 + \
                           ((self.friction_param_1 - self.friction_param_0) / (0.16 - 0.002) * (velocities - 0.002) + self.friction_param_0) * flag_1 \
                           + (self.friction_param_1 + self.friction_param_3 * (velocities - 0.16)) * flag_2 \
                           + ((self.friction_param_2 + self.friction_param_0) / (-0.16 + 0.002) * (velocities + 0.002) - self.friction_param_0) * flag_3 \
                           + (self.friction_param_2 + self.friction_param_4 * (velocities + 0.16)) * flag_4

        self.torques_friction = torques - torques_friction

        return self.torques_friction.clone()


class ActuatorFeature802030(ActuatorFeature):
    def __init__(self, num_envs, num_actions, device=None):
        super(ActuatorFeature802030, self).__init__(num_envs, num_actions, device, 1.2766, 12.13208)

    def reset(self, env_ids):
        super(ActuatorFeature802030, self).reset(env_ids)

        self.friction_param_0[env_ids] = torch_rand_float(3.7, 6.6, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_1[env_ids] = torch_rand_float(3.3, 5.0, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_2[env_ids] = torch_rand_float(-5.0, -3.3, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_3[env_ids] = torch_rand_float(0.7, 0.9, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_4[env_ids] = torch_rand_float(0.7, 0.9, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)


class ActuatorFeature601750(ActuatorFeature):
    def __init__(self, num_envs, num_actions, device=None):
        super(ActuatorFeature601750, self).__init__(num_envs, num_actions, device, 0.2419, 10.4578)

    def reset(self, env_ids):
        super(ActuatorFeature601750, self).reset(env_ids)

        self.friction_param_0[env_ids] = torch_rand_float(1.2, 2.75, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_1[env_ids] = torch_rand_float(1.0, 1.55, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_2[env_ids] = torch_rand_float(-1.55, -1.0, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_3[env_ids] = torch_rand_float(0.4, 0.65, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_4[env_ids] = torch_rand_float(0.4, 0.65, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)


class ActuatorFeature1307E(ActuatorFeature):
    def __init__(self, num_envs, num_actions, device=None):
        super(ActuatorFeature1307E, self).__init__(num_envs, num_actions, device, 0.91, 11.28)

    def reset(self, env_ids):
        super(ActuatorFeature1307E, self).reset(env_ids)

        self.friction_param_0[env_ids] = torch_rand_float(1.9, 3.3, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_1[env_ids] = torch_rand_float(1.15, 2.0, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_2[env_ids] = torch_rand_float(-2.0, -1.3, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_3[env_ids] = torch_rand_float(0.14, 0.18, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_4[env_ids] = torch_rand_float(0.14, 0.18, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)


class ActuatorFeature3611100(ActuatorFeature):
    def __init__(self, num_envs, num_actions, device=None):
        super(ActuatorFeature3611100, self).__init__(num_envs, num_actions, device, 1, 1)

    def reset(self, env_ids):
        super(ActuatorFeature3611100, self).reset(env_ids)

        self.friction_param_0[env_ids] = torch_rand_float(0.25, 1.25, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_1[env_ids] = torch_rand_float(0.2, 1.0, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_2[env_ids] = torch_rand_float(-1.0, -0.2, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_3[env_ids] = torch_rand_float(0.14, 0.18, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
        self.friction_param_4[env_ids] = torch_rand_float(0.14, 0.18, (len(env_ids), self.num_joints_use), device=self.device).squeeze(1)
