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

        self.landmarks = np.zeros((num_landmarks, 2))
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
        self.Wm = np.zeros(n_sigma)
        self.Wc = np.zeros(n_sigma)

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
        return new_state

    def measurement_model(self, state, landmark_id):
        """
        Predict measurement given current state and landmark

        STUDENT TODO:
        1. Calculate relative position: landmark - robot position
        2. Transform to robot frame using robot orientation
        3. Return relative position in robot frame
        """
        return np.zeros(2)

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
        print("UKF Predict: TODO - Implement prediction step")

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
        if not landmark_observations:
            return
        print("UKF Update: TODO - Implement measurement update step")

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
