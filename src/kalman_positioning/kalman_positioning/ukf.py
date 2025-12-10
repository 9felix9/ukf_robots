import numpy as np

np.set_printoptions(suppress=True, precision=6)


class UKF:
    """
    Unscented Kalman Filter für 5D Roboterzustand:
        x = [x, y, theta, vx, vy]

    Messungen:
        Landmark-Positionen im Robot-Frame: z = [lx_robot, ly_robot]

    Diese Version kombiniert:
        - Deine Struktur
        - Die korrekten Sigma-Point-Gewichte deines Kommilitonen
        - Sicheres Cholesky (Fallback auf Jitter)
        - Korrekte Mittelwertbildung & Kovarianzberechnung
        - Korrekte Messinnovation
    """

    # UKF-Standardparameter (Van der Merwe)
    def __init__(self, process_noise_xy, process_noise_theta, measurement_noise_xy, num_landmarks):
        self.nx = 5          # state dimension
        self.nz = 2          # measurement dimension

        # Landmarks als Tabelle: [id, x, y]
        self.landmarks = np.zeros((num_landmarks, 3))

        # UKF Parameter
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0

        # Sigma scaling
        self.lambda_ = self.alpha**2 * (self.nx + self.kappa) - self.nx
        self.gamma = np.sqrt(self.nx + self.lambda_)

        # State
        self.x_ = np.zeros(self.nx)
        self.P_ = np.eye(self.nx)

        # Prozessrauschen Q (5×5)
        self.Q_ = np.diag([
            process_noise_xy,
            process_noise_xy,
            process_noise_theta,
            0.0, 0.0
        ])

        # Messrauschen R (2×2)
        self.R_ = np.diag([measurement_noise_xy, measurement_noise_xy])

        # Gewichtung
        self.Wm = np.zeros(2*self.nx + 1)
        self.Wc = np.zeros(2*self.nx + 1)

        self.Wm[0] = self.lambda_ / (self.nx + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

        wi = 1.0 / (2.0 * (self.nx + self.lambda_))
        self.Wm[1:] = wi
        self.Wc[1:] = wi

    # -------------------------------------------------------------
    # Sigma-Punkte
    # -------------------------------------------------------------
    def generate_sigma_points(self, mean, cov):
        n_sigma = 2*self.nx + 1
        Chi = np.zeros((self.nx, n_sigma))

        # Fallback falls Matrix nicht PD ist
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov + 1e-6*np.eye(self.nx))

        Chi[:, 0] = mean.copy()

        for i in range(self.nx):
            offset = self.gamma * L[:, i]
            Chi[:, 1+i] = mean + offset
            Chi[:, 1+self.nx+i] = mean - offset

        return Chi

    # -------------------------------------------------------------
    # Prozessmodell
    # -------------------------------------------------------------
    def process_model(self, state, dt, dx, dy, dtheta):
        new_state = state.copy()

        new_state[0] = state[0] + dx
        new_state[1] = state[1] + dy
        new_state[2] = self.normalize_angle(state[2] + dtheta)

        new_state[3] = dx/dt
        new_state[4] = dy/dt
        return new_state

    # -------------------------------------------------------------
    # Messmodell: Landmark im Robot-Frame
    # -------------------------------------------------------------
    def measurement_model(self, state, landmark_id):
        # Landmark lookup
        mask = self.landmarks[:, 0] == landmark_id
        lm = self.landmarks[mask]

        if lm.size == 0:
            raise ValueError(f"Landmark ID {landmark_id} not found in UKF.landmarks!")

        lx, ly = lm[0, 1], lm[0, 2]

        x, y, theta = state[0], state[1], state[2]

        dx = lx - x
        dy = ly - y

        R = np.array([
            [ np.cos(theta),  np.sin(theta)],
            [-np.sin(theta),  np.cos(theta)]
        ])

        return R @ np.array([dx, dy])

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------
    def predict(self, dt, dx, dy, dtheta):
        X_sigma = self.generate_sigma_points(self.x_, self.P_)

        X_pred = np.zeros_like(X_sigma)

        for i in range(X_sigma.shape[1]):
            X_pred[:, i] = self.process_model(X_sigma[:, i], dt, dx, dy, dtheta)

        # Mittelwert
        x_pred = np.sum(self.Wm * X_pred, axis=1)

        # Kovarianz
        P_pred = np.zeros((self.nx, self.nx))
        for i in range(X_pred.shape[1]):
            diff = X_pred[:, i] - x_pred
            diff[2] = self.normalize_angle(diff[2])
            P_pred += self.Wc[i] * np.outer(diff, diff)

        P_pred += self.Q_

        self.x_ = x_pred
        self.P_ = P_pred

    # -------------------------------------------------------------
    # Update für eine Landmark-Messung
    # -------------------------------------------------------------
    def update(self, obs_tuple):
        """
        obs_tuple = (obs_x_world, obs_y_world, landmark_id)
        Simulator liefert die Messung im WORLD-Frame.
        UKF erwartet Messung im ROBOT-Frame.
        Deshalb transformieren wir zuerst WORLD → ROBOT.
        """

        obs_x_w, obs_y_w, lm_id = obs_tuple
        lm_id = int(lm_id)

        # --- WORLD → ROBOT Transformation ---
        robot_x = self.x_[0]
        robot_y = self.x_[1]
        robot_theta = self.x_[2]

        dx = obs_x_w - robot_x
        dy = obs_y_w - robot_y

        R = np.array([
            [ np.cos(robot_theta),  np.sin(robot_theta)],
            [-np.sin(robot_theta),  np.cos(robot_theta)]
        ])

        z_meas = R @ np.array([dx, dy])   # Messung jetzt im ROBOT-FRAME

        # --- UKF UPDATE beginnt hier ---
        X_sigma = self.generate_sigma_points(self.x_, self.P_)

        Z_sigma = np.zeros((self.nz, 2*self.nx + 1))
        for i in range(Z_sigma.shape[1]):
            Z_sigma[:, i] = self.measurement_model(X_sigma[:, i], lm_id)

        z_pred = np.sum(self.Wm * Z_sigma, axis=1)

        S = np.zeros((self.nz, self.nz))
        Tc = np.zeros((self.nx, self.nz))

        for i in range(Z_sigma.shape[1]):
            z_diff = Z_sigma[:, i] - z_pred
            x_diff = X_sigma[:, i] - self.x_

            x_diff[2] = self.normalize_angle(x_diff[2])

            S += self.Wc[i] * np.outer(z_diff, z_diff)
            Tc += self.Wc[i] * np.outer(x_diff, z_diff)

        S += self.R_

        K = Tc @ np.linalg.inv(S)
        innovation = z_meas - z_pred

        self.x_ = self.x_ + K @ innovation
        self.x_[2] = self.normalize_angle(self.x_[2])

        self.P_ = self.P_ - K @ S @ K.T

    # -------------------------------------------------------------
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2*np.pi) - np.pi
