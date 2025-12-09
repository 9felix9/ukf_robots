import numpy as np

class UKF:
    def __init__(self, process_noise_xy, process_noise_theta, measurement_noise_xy, num_landmarks):
        """
        Initialize the Unscented Kalman Filter

        STUDENT TODO:
        1. Initialize filter parameters (alpha, beta, kappa, lambda)
        2. Initialize state vector x_ with zeros
        3. Initialize state covariance matrix P_
        4. Set process noise covariance Q_
        5. Set measurement noise covariance R_
        6. Calculate sigma point weights for mean and covariance
        """
        self.nx = 5
        self.nz = 2

        self.landmarks = np.zeros((num_landmarks, 3))
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.x_ = np.zeros((self.nx,), dtype=float).T
        self.P_ = np.identity(self.nx)
        self.Q_ = np.diag(np.array([process_noise_xy, process_noise_xy, process_noise_theta, 0, 0])) # covarianz matrix für normalverteilung
        self.R_ = np.diag([measurement_noise_xy, measurement_noise_xy]) # ebenfalls covarianz matrix

        # Calculate UKF parameters: λ, γ, and weights W^m_i, W^c_i
        self.lamb = self.alpha**2 * (self.nx + self.kappa) - self.nx
        self.gamma = np.sqrt(self.nx + self.lamb)

        # TODO weights
        n_sigma = 2*self.nx + 1
        self.Wm = np.zeros(n_sigma) # weights mean
        self.Wc = np.zeros(n_sigma) # weights covariance

        # Weight for the mean of the first sigma point
        self.Wm[0] = self.lamb / (self.nx + self.lamb)

        # Weight for the covariance of the first sigma point
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

        # Remaining 2n weights
        for i in range(1, n_sigma):
            w = self.lamb / 2 * (self.nx + self.lamb)
            self.Wm[i] = w
            self.Wc[i] = w

        # print("UKF Constructor: TODO - Implement filter initialization")

    def generate_sigma_points(self, mean, cov):
        """
        Generate sigma points from mean and covariance

        STUDENT TODO:
        1. Start with the mean as the first sigma point
        2. Compute Cholesky decomposition of covariance
        3. Generate 2*n symmetric sigma points around the mean
        """

        # 1. first Sigma point
        Cholesky= np.linalg.cholesky(cov) # is like the sqrt of the matrix
        Chi = np.zeros((self.nx, 2*self.nx + 1))
        Chi[:,0] = mean
        for i in range(1 ,self.nx): 
            Chi[:, i] = mean + np.sqrt(self.nx + self.lamb) * Cholesky[:,i]
            Chi[:, i + self.nx] = mean - np.sqrt(self.nx + self.lamb) * Cholesky[:,i]

        return Chi

    def process_model(self, state, dt, dx, dy, dtheta):
        """
        Apply motion model to a state vector

        STUDENT TODO:
        1. Updates position: x' = x + dx, y' = y + dy
        2. Updates orientation: theta' = theta + dtheta (normalized)
        3. Updates velocities: vx' = dx/dt, vy' = dy/dt
        """
        new_state = state
        
        # x'
        new_state[0] = state[0] + dx

        # y'
        new_state[1] = state[1] + dy
        
        # theta'
        new_state[2] = self.normalize_angle(state[2] + dtheta) 
        
        # vx' (geschwinidigkeit x) 
        new_state[3] = dx/dt
        
        # xy' (geschwindigkeit y)
        new_state[4] = dy/dt

        return new_state

    def measurement_model(self, state, landmark_id):
        """
        Predict measurement given current state and landmark.

        Returns landmark position in robot frame.
        """

        # Robot state
        x = state[0]
        y = state[1]
        theta = state[2]

        # 1. Find landmark position (in world coordinates)
        mask = self.landmarks[:, 0] == landmark_id
        landmark = self.landmarks[mask]

        if landmark.size == 0:
            raise ValueError(f"Landmark ID {landmark_id} not found!")

        lx = landmark[0, 1]
        ly = landmark[0, 2]

        # 2. Relative position in world frame
        dx = lx - x
        dy = ly - y
        relative = np.array([dx, dy])

        # 3. Rotation: world -> robot frame
        # R(-theta)
        R = np.array([
            [ np.cos(theta),  np.sin(theta)],
            [-np.sin(theta),  np.cos(theta)]
        ])

        robot_frame = R @ relative

        return robot_frame


    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def predict(self, dt, dx, dy, dtheta):
        """
        Kalman Filter Prediction Step (Time Update)

        STUDENT TODO:
        1. Generate sigma points from current state and covariance
        2. Propagate each sigma point through motion model
        3. Calculate mean and covariance of predicted sigma points
        4. Add process noise
        5. Update state and covariance estimates
        """
        mean = self.x_
        Sigma_points = self.generate_sigma_points(mean, self.P_)
        Transformed_sigma_points = np.zeros(np.shape(Sigma_points))
        for i in range(np.shape(Sigma_points)[1]): 
            sigma_point_state = np.copy(Sigma_points[:,i])
            Transformed_sigma_points[:,i] = self.process_model(sigma_point_state, dt, dx, dy, dtheta)
        # self.Wm = (11,) --> (1, 11)
        # compatible with Transformed_sigma_points
        x_pred = np.sum(self.Wm[np.newaxis, :] * Transformed_sigma_points, axis=1)
        
        # weighted covariance 
        P_pred = np.zeros_like(self.P_)
        
        for i in range(2*self.nx+1):
            # 5x1 @ 1x5 = 5x5
            x_sigma_pred = Transformed_sigma_points[:,i]
            diff = x_sigma_pred[:, np.newaxis] - x_pred[:, np.newaxis]
            P_pred += self.Wc[i] * (diff @ diff.T)
        
        P_pred += self.Q_

        # store predicted results
        self.x_ = x_pred
        self.P_ = P_pred

    def update(self, landmark_observations):
        """
        Kalman Filter Update Step (Measurement Update)

        STUDENT TODO:
        1. Generate sigma points
        2. Transform through measurement model
        3. Calculate predicted measurement mean
        4. Calculate measurement and cross-covariance
        5. Compute Kalman gain
        6. Update state with innovation
        7. Update covariance
        """
        for obs in landmark_observations:
            lm_id = obs[0]
            z_measured_world = np.array([obs[1], obs[2]])

            # 1️⃣ Sigma-Punkte aus aktuellem Zustand
            X_sigma = self.generate_sigma_points(self.x_, self.P_)

            # 2️⃣ Sigma-Punkte durch Messmodell
            Z_sigma = np.zeros((self.nz, 2*self.nx + 1))
            for i in range(2*self.nx + 1):
                Z_sigma[:, i] = self.measurement_model(X_sigma[:, i], lm_id)

            # 3️⃣ Messvorhersage (gewichteter Mittelwert)
            z_pred = np.sum(self.Wm[np.newaxis, :] * Z_sigma, axis=1)

            # 4️⃣ Deltas (Differenzen)
            z_diff = Z_sigma - z_pred[:, np.newaxis]
            x_diff = X_sigma - self.x_[:, np.newaxis]

            # Winkel normalisieren
            x_diff[2, :] = np.vectorize(self.normalize_angle)(x_diff[2, :])

            # 5️⃣ Messkovarianz (S) & Kreuzkovarianz (Tc)
            S = z_diff @ np.diag(self.Wc) @ z_diff.T + self.R_
            Tc = x_diff @ np.diag(self.Wc) @ z_diff.T
            
            # Kalman gain
            K = Tc @ np.linalg.inv(S)
            
            # calculate innovation
            z_pred_for_state = self.measurement_model(self.x_, lm_id)
            innovation = z_measured_world - z_pred_for_state
            
            self.x_ = self.x_ + (K @ innovation).flatten()
            self.P_ = self.P_ - K @ S @ K.T

if __name__ == "__main__": 
    # Test parameters for UKF initialization
    process_noise_xy = 0.1
    process_noise_theta = 0.05
    measurement_noise_xy = 0.2
    num_landmarks = 10

    # Instantiate UKF with test parameters
    ukf = UKF(process_noise_xy, process_noise_theta, measurement_noise_xy, num_landmarks)
    
    
    print("\n--- Testing generate_sigma_points() ---")

    # simple test mean and covariance
    mean = np.array([1.0, 2.0, 0.5, 0.0, 0.0])
    cov = np.eye(5) * 0.1   # simple diagonal covariance

    sigma = ukf.generate_sigma_points(mean, cov)

    print("Sigma points shape:", sigma.shape)
    print("Sigma points:\n", sigma)

    # Basic checks
    print("\nCheck first sigma point (should be the mean):")
    print(sigma[:,0])

    print("\nCheck plus/minus sigma point distance along first dimension:")
    print("plus:", sigma[0,1], "minus:", sigma[0,1 + ukf.nx])

    print("\nIf the implementation is correct,")
    print("- sigma has shape (5, 11)")
    print("- sigma[:,0] equals the mean")
    print("- plus and minus sigma points differ symmetrically.")


    print("\n================= UKF PREDICT() TEST =================")

    # Reset UKF state so test is clean
    ukf.x_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    ukf.P_ = np.eye(5) * 0.1

    dt = 1.0
    dx = 1.0
    dy = 0.5
    dtheta = 0.1

    print("\n--- BEFORE PREDICT ---")
    print("State x_:\n", ukf.x_)
    print("Covariance P_:\n", ukf.P_)

    # Show sigma points before prediction
    sigma_before = ukf.generate_sigma_points(ukf.x_, ukf.P_)
    print("\nSigma points BEFORE motion:")
    print(sigma_before)

    # Perform prediction step
    print("\nRunning predict() ...")
    ukf.predict(dt, dx, dy, dtheta)

    print("\n--- AFTER PREDICT ---")
    print("State x_:\n", ukf.x_)
    print("Covariance P_:\n", ukf.P_)

    # Show sigma points after motion model
    sigma_after = np.zeros_like(sigma_before)
    for i in range(sigma_before.shape[1]):
        sigma_after[:, i] = ukf.process_model(sigma_before[:, i], dt, dx, dy, dtheta)

    print("\nSigma points AFTER motion model:")
    print(sigma_after)
    print("\nYou should now see how sigma points moved & how state/covariance changed!")
