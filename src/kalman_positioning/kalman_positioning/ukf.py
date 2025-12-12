import numpy as np

np.set_printoptions(suppress=True, precision=6)


class UKF:
    """
    Unscented Kalman Filter for 2D robot localization.

    State vector:
        x = [pos_x, pos_y, yaw, vel_x, vel_y]

    Measurement:
        z = [landmark_dx, landmark_dy]   (expressed in robot frame)
    """

    def __init__(self, process_noise_xy, process_noise_theta, measurement_noise_xy, num_landmarks):
        self.state_dim = 5
        self.meas_dim = 2

        # Landmark table: [id, x_world, y_world]
        self.landmarks = np.zeros((num_landmarks, 3))

        # ----------------------------------------------------------------------
        # UKF Parameters (Van der Merwe)
        # ----------------------------------------------------------------------
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0

        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.gamma = np.sqrt(self.state_dim + self.lambda_)

        # State and covariance
        self.state = np.zeros(self.state_dim)
        self.cov = np.eye(self.state_dim)

        # ----------------------------------------------------------------------
        # Process noise matrix  (mathematically: Q)
        # ----------------------------------------------------------------------
        self.process_noise = np.diag([
            process_noise_xy,
            process_noise_xy,
            process_noise_theta,
            0.0,
            0.0
        ])

        # ----------------------------------------------------------------------
        # Measurement noise matrix  (mathematically: R)
        # ----------------------------------------------------------------------
        self.measurement_noise = np.diag([
            measurement_noise_xy,
            measurement_noise_xy
        ])

        # ----------------------------------------------------------------------
        # Sigma-point weights
        # ----------------------------------------------------------------------
        count_sigma = 2*self.state_dim + 1
        self.w_m = np.full(count_sigma, 1.0 / (2*(self.state_dim + self.lambda_)))
        self.w_c = self.w_m.copy()

        self.w_m[0] = self.lambda_ / (self.state_dim + self.lambda_)
        self.w_c[0] = self.w_m[0] + (1 - self.alpha**2 + self.beta)


    # ======================================================================
    # Sigma point generation
    # ======================================================================
    def generate_sigma_points(self):
        n = self.state_dim
        sigma_count = 2*n + 1

        # Cholesky decomposition with fallback
        try:
            L = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(self.cov + 1e-6*np.eye(n))

        sigma = np.zeros((n, sigma_count))
        sigma[:, 0] = self.state

        for i in range(n):
            offset = self.gamma * L[:, i]
            sigma[:, 1+i] = self.state + offset
            sigma[:, 1+n+i] = self.state - offset

        return sigma


    # ======================================================================
    # Process model: integrates odometry motion increments
    # ======================================================================
    def process_model(self, state, dt, dx, dy, dtheta):
        new = state.copy()

        new[0] += dx
        new[1] += dy
        new[2] = self.normalize_angle(state[2] + dtheta)

        new[3] = dx / dt
        new[4] = dy / dt

        return new


    # ======================================================================
    # Measurement model: expected landmark position in robot frame
    # ======================================================================
    def measurement_model(self, state, landmark_id):
        lm = self.landmarks[self.landmarks[:, 0] == landmark_id]

        if lm.size == 0:
            raise ValueError(f"Landmark {landmark_id} not found")

        lx, ly = lm[0, 1], lm[0, 2]
        x, y, yaw = state[0], state[1], state[2]

        dx = lx - x
        dy = ly - y

        R = np.array([
            [np.cos(yaw),  np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])

        return R @ np.array([dx, dy])


    # ======================================================================
    # Prediction step
    # ======================================================================
    def predict(self, dt, dx, dy, dtheta):
        sigma = self.generate_sigma_points()

        sigma_pred = np.zeros_like(sigma)
        for i in range(sigma.shape[1]):
            sigma_pred[:, i] = self.process_model(sigma[:, i], dt, dx, dy, dtheta)

        mean = np.sum(self.w_m * sigma_pred, axis=1)

        cov = np.zeros((self.state_dim, self.state_dim))
        for i in range(sigma_pred.shape[1]):
            diff = sigma_pred[:, i] - mean
            diff[2] = self.normalize_angle(diff[2])
            cov += self.w_c[i] * np.outer(diff, diff)

        cov += self.process_noise

        self.state = mean
        self.cov = cov


    # ======================================================================
    # Update step
    # ======================================================================
    def update(self, obs):
        obs_x, obs_y, lm_id = obs
        z_meas = np.array([obs_x, obs_y])

        sigma = self.generate_sigma_points()

        meas_sigma = np.zeros((self.meas_dim, sigma.shape[1]))
        for i in range(sigma.shape[1]):
            meas_sigma[:, i] = self.measurement_model(sigma[:, i], lm_id)

        z_pred = np.sum(self.w_m * meas_sigma, axis=1)

        S = self.measurement_noise.copy()
        Tc = np.zeros((self.state_dim, self.meas_dim))

        for i in range(sigma.shape[1]):
            z_diff = meas_sigma[:, i] - z_pred
            x_diff = sigma[:, i] - self.state

            x_diff[2] = self.normalize_angle(x_diff[2])

            S += self.w_c[i] * np.outer(z_diff, z_diff)
            Tc += self.w_c[i] * np.outer(x_diff, z_diff)

        K = Tc @ np.linalg.inv(S)

        innovation = z_meas - z_pred

        self.state = self.state + K @ innovation
        self.state[2] = self.normalize_angle(self.state[2])

        self.cov = self.cov - K @ S @ K.T


    # ======================================================================
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2*np.pi) - np.pi
