import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from IPython.display import YouTubeVideo
from scipy.stats import norm


def kalman(ID_xyz):
    # initial uncertainty
    P = 100.0 * np.eye(9)

    ############### DYNAMIC MATRIX #############################################
    dt = 0.04  # Time Step between Filter Steps 1/25

    A = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 1 / 2.0 * dt ** 2, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0, 1 / 2.0 * dt ** 2, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0, 1 / 2.0 * dt ** 2],
                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, dt, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, dt],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    print(A.shape)

    ################# MEASUREMENT MATRIX #####################################
    H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    print(H, H.shape)

    ################ MEASUREMENT NOISE COVARIANCE MATRIX R ##################

    rp = 1.0 ** 2  # Noise of Position Measurement
    R = np.matrix([[rp, 0.0, 0.0],
                   [0.0, rp, 0.0],
                   [0.0, 0.0, rp]])
    print(R, R.shape)

    ####### PROCESS NOISE COVARIANCE MATRIX Q ###############################

    sj = 0.1

    Q = np.matrix([[(dt ** 6) / 36, 0, 0, (dt ** 5) / 12, 0, 0, (dt ** 4) / 6, 0, 0],
                   [0, (dt ** 6) / 36, 0, 0, (dt ** 5) / 12, 0, 0, (dt ** 4) / 6, 0],
                   [0, 0, (dt ** 6) / 36, 0, 0, (dt ** 5) / 12, 0, 0, (dt ** 4) / 6],
                   [(dt ** 5) / 12, 0, 0, (dt ** 4) / 4, 0, 0, (dt ** 3) / 2, 0, 0],
                   [0, (dt ** 5) / 12, 0, 0, (dt ** 4) / 4, 0, 0, (dt ** 3) / 2, 0],
                   [0, 0, (dt ** 5) / 12, 0, 0, (dt ** 4) / 4, 0, 0, (dt ** 3) / 2],
                   [(dt ** 4) / 6, 0, 0, (dt ** 3) / 2, 0, 0, (dt ** 2), 0, 0],
                   [0, (dt ** 4) / 6, 0, 0, (dt ** 3) / 2, 0, 0, (dt ** 2), 0],
                   [0, 0, (dt ** 4) / 6, 0, 0, (dt ** 3) / 2, 0, 0, (dt ** 2)]]) * sj ** 2

    print(Q.shape)

    ############### DISTURBANCE CONTROL MATRIC B ########################
    B = np.matrix([[0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0],
                   [0.0]])
    print(B, B.shape)

    ######## CONTROL INPUT u #######################################
    u = 0.0

    ########## Identity matrix #############################
    I = np.eye(9)
    print(I, I.shape)

    #################### Measurements ##########################
    X = []
    Y = []
    Z = []
    for xyz in ID_xyz:
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        X.append(x)
        Y.append(y)
        Z.append(z)

    print("X: ", X)
    print("Y: ", Y)
    print("Z: ", Z)

    ########## ADD NOISE TO THE REAL POSITION ################################
    # T = 1.0 # s measuremnt time
    # m = int(T/dt) # number of measurements
    m = int(len(X))

    sp = 0.1  # Sigma for position noise

    # print("SPPPP: ", sp * (np.random.randn(m)))

    print("len sp: ", len(sp * (np.random.randn(m))))
    print("len Xr: ", len(X))
    print("X: ", len(X))

    Xm = X + sp * (np.random.randn(m))
    Ym = Y + sp * (np.random.randn(m))
    Zm = Z + sp * (np.random.randn(m))

    measurements = np.vstack((Xm, Ym, Zm))
    print(measurements.shape)

    #### INITIAL STATE ############################
    x = np.matrix([0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0, -9.81]).T
    print(x, x.shape)

    ### Plotting
    # Preallocation for Plotting
    xt = []
    yt = []
    zt = []
    dxt = []
    dyt = []
    dzt = []
    ddxt = []
    ddyt = []
    ddzt = []
    Zx = []
    Zy = []
    Zz = []
    Px = []
    Py = []
    Pz = []
    Pdx = []
    Pdy = []
    Pdz = []
    Pddx = []
    Pddy = []
    Pddz = []
    Kx = []
    Ky = []
    Kz = []
    Kdx = []
    Kdy = []
    Kdz = []
    Kddx = []
    Kddy = []
    Kddz = []

    ##### KALMAN FILTER ####################

    for filterstep in range(m):
        # Model the direction switch, when hitting the plate
        # if x[2] < 0.01 and not hitplate:
        #        x[5] = -x[5]
        #        hitplate = True

        # Time Update (Prediction)
        # ========================
        # Project the state ahead
        x = A * x + B * u

        # Project the error covariance ahead
        P = A * P * A.T + Q

        # Measurement Update (Correction)
        # ===============================
        # Compute the Kalman Gain
        S = H * P * H.T + R
        K = (P * H.T) * np.linalg.pinv(S)

        # Update the estimate via z
        Z = measurements[:, filterstep].reshape(H.shape[0], 1)
        y = Z - (H * x)  # Innovation or Residual
        x = x + (K * y)

        # Update the error covariance
        P = (I - (K * H)) * P

        # Save states for Plotting
        xt.append(float(x[0]))
        yt.append(float(x[1]))
        zt.append(float(x[2]))
        dxt.append(float(x[3]))
        dyt.append(float(x[4]))
        dzt.append(float(x[5]))
        ddxt.append(float(x[6]))
        ddyt.append(float(x[7]))
        ddzt.append(float(x[8]))

        Zx.append(float(Z[0]))
        Zy.append(float(Z[1]))
        Zz.append(float(Z[2]))
        Px.append(float(P[0, 0]))
        Py.append(float(P[1, 1]))
        Pz.append(float(P[2, 2]))
        Pdx.append(float(P[3, 3]))
        Pdy.append(float(P[4, 4]))
        Pdz.append(float(P[5, 5]))
        Pddx.append(float(P[6, 6]))
        Pddy.append(float(P[7, 7]))
        Pddz.append(float(P[8, 8]))
        Kx.append(float(K[0, 0]))
        Ky.append(float(K[1, 0]))
        Kz.append(float(K[2, 0]))
        Kdx.append(float(K[3, 0]))
        Kdy.append(float(K[4, 0]))
        Kdz.append(float(K[5, 0]))
        Kddx.append(float(K[6, 0]))
        Kddy.append(float(K[7, 0]))
        Kddz.append(float(K[8, 0]))

    ############## PLOT: Position in x/y plane ###########
    '''
    fig = plt.figure(figsize=(16, 9))

    plt.plot(xt, yt, label='Kalman filter')
    plt.scatter(Xm, Ym, label='Measurement', c='gray', s=30)
    # plt.plot(X, Y, label='Real')
    plt.legend(loc='best', prop={'size': 22})
    # plt.axhline(0, color='k')
    plt.axis('equal')
    plt.xlabel('x ($mm$)')
    plt.ylabel('y ($mm$)')
    # plt.ylim(0, 2);
    plt.show()
    '''
    print("xt_0: ", xt[0])
    print("xt: ", xt[m - 1])
    print("XM_0: ", Xm[0])
    print("XM: ", Xm[m - 1])

    '''
        print("XM: ", Xm)
        print("yt: ", yt)
        print("YM: ", Ym)
        print("zt: ", zt)
        print("ZM: ", Zm)
        '''
    # Estimates for start position
    x_0 = xt[0]
    y_0 = yt[0]
    z_0 = zt[0]

    # Estimates for end position
    x_1 = xt[m - 1]
    y_1 = yt[m - 1]
    z_1 = zt[m - 1]

    return x_0, y_0, z_0, x_1, y_1, z_1
