import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import matrix_rank
from scipy.linalg import inv


def fit_trend_plus_freq(time, data, freq, dataE, polydeg, fixVal):
    """
    The function fits mean, trend, harmonics, and polynomials to the data,
    depending on the input.

    Input:
        time ... time (or x) vector
        data ... data (or y) vector
        freq ... f x 1 array with the frequencies (relative to time vector)
        dataE ... data error vector (y_e) (if none -> empty array)
        polydeg ... integer selecting the polynomial degree (default 1)

    Output:
        p = parameter vector [polynomial(mean, trend, ...) harmonic information, {amplitude and phase information}]
        pE = corresponding covariance matrix
        sig = posteriori sigma0
        amplPhas = if chosen, amplitude and phase information will be
                   provided [ampl phase amplE(*sig) phaseE(*sig)]
    """
    if polydeg is None:
        polydeg = 1

    amplPhas = np.full((len(freq), 4), np.nan)

    idz = [1]

    # construct error matrix
    if dataE is None:
        E = np.eye(len(time))
    # else:
    #     dataE = dataE[idn]
    #     E = np.diag((1.0 / dataE ** 2))

    # construct design matrix, the first matrix is for the polynomial part, the next correspond to the cosine and
    # sine terms for each frequency
    A = np.zeros((len(data), polydeg + 1 + len(freq) * 2))
    cc = 0
    for j in range(polydeg + 1):
        A[:, j] = time ** j
        cc += 1
    ccF = cc
    for j in range(len(freq)):
        A[:, cc:cc + 2] = np.column_stack((np.cos(freq[j] * 2 * np.pi * time), np.sin(freq[j] * 2 * np.pi * time)))
        cc += 2

    # check if matrix is full rank
    if matrix_rank(A) != A.shape[1]:
        p = np.full((polydeg + 1 + len(freq) * 2,), np.nan)
        pE = np.full((polydeg + 1 + len(freq) * 2, polydeg + 1 + len(freq) * 2), np.nan)
        sig = np.nan
        return p, pE, sig, amplPhas

    # solve the normal equations
    N = A.T @ E @ A
    n = A.T @ E @ data

    if not np.isnan(N[0, 0]):
        pE = inv(N)
        p = pE @ n
    else:
        p = np.full((A.shape[1],), np.nan)
        pE = np.full_like(N, np.nan)
        sig = np.nan

    # calculate posteriori sigma0 and covariance matrix of the parameters
    v = A @ p - data
    sig = (v.T @ E @ v) / (A.shape[0] - A.shape[1])
    pE = pE * sig

    # add fixed values to the parameter vector and covariance matrix
    if len(idz) > 1:
        hv = np.full((p.shape[0], len(idz)), np.nan)
        hv[:, idz] = p
        p = hv

    # calculate amplitude and phase information
    cch = ccF
    amplPhas = np.zeros((len(freq), 4))
    for j in range(len(freq)):
        freqMod = freq[j] * 2 * np.pi
        ampl = np.sqrt(p[cch] ** 2 + p[cch + 1] ** 2)
        ph = np.arctan2(p[cch + 1], p[cch]) / freqMod
        prop = np.array([[p[cch] / ampl, p[cch + 1] / ampl],
                         [p[cch + 1] / (p[cch] ** 2 * (1 + (p[cch + 1] / p[cch]) ** 2) * freqMod),
                          1 / (p[cch] * (1 + (p[cch + 1] / p[cch]) ** 2) * freqMod)]])
        covnew = pE[cch:cch + 2, cch:cch + 2]
        covnew = prop @ (covnew @ prop.T)
        amplPhas[j, :] = [ampl, ph, np.sqrt(covnew[0, 0]), np.sqrt(covnew[1, 1])]
        cch += 2

    return p, pE, sig, amplPhas


def reconstruct_signal(time, p, freq, polydeg):
    """
    Reconstructs the signal using the parameters from the fit.

    Parameters:
    time (array-like): Time vector.
    p (array-like): Parameter vector from the fit.
    freq (array-like): Array of frequencies.
    polydeg (int): Degree of the polynomial.

    Returns:
    y (numpy array): Reconstructed signal.
    """

    # Compute the mean of the time vector
    time_mean = np.mean(time)

    # Initialize y with the polynomial part
    y = np.zeros_like(time)
    y += p[0]

    for i in range(1, polydeg + 1):
        y += p[i] * (time - time_mean) ** i

    # Add the harmonic components
    cc = polydeg + 1
    for j in range(len(freq)):
        freq_mod = 2 * np.pi * freq[j]
        y += p[cc] * np.cos(freq_mod * (time - time_mean)) + p[cc + 1] * np.sin(freq_mod * (time - time_mean))
        cc += 2

    return y


if __name__ == "__main__":
    # read in station data for station 1
    time = []
    data = []
    with open("../../data/Meeresdaten_simuliert/data/2010.rlrdata", "r") as file:
        for line in file:
            split_line = line.split(";")
            date = float(split_line[0].strip())
            sea_level = float(split_line[1].strip())
            flag = split_line[3].strip()
            time.append(date)
            data.append(sea_level)
    time = np.array(time)
    data = np.array(data)
    freq = np.array([1, 2])
    dataE = None
    polydeg = 1
    fixVal = None
    p, pE, sig, amplPhas = fit_trend_plus_freq(time, data, freq, dataE, polydeg, fixVal)
    # print(p)
    y = reconstruct_signal(time, p, freq, polydeg)

    complete_sum = 0
    mean_centered_y = []
    for y_i in y:
        complete_sum += y_i
    avg = complete_sum / len(y)
    for y_i in range(len(y)):
        mean_centered_y.append(y[y_i] - avg)

    complete_sum = 0
    mean_centered_data = []
    for i in data:
        complete_sum += i
    avg = complete_sum / len(data)
    for i in range(len(data)):
        mean_centered_data.append(data[i] - avg)

    season_controlled = []
    for date in range(len(time)):
        if abs(mean_centered_data[date] - mean_centered_y[date]) > 0.05:
            season_controlled.append(-99999.0)
        else:
            season_controlled.append(mean_centered_data)

    fig, ax = plt.subplots()
    ax.plot(time, mean_centered_data, color="green")
    ax.plot(time, mean_centered_y, color="blue", linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Sea level')
    plt.savefig("../output/season_corrected/station1.pdf")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(time, season_controlled, color="green")
    plt.xlabel("Time")
    plt.ylabel("Sea level")
    plt.savefig("../output/season_corrected/station1_controlled_5cm")
    plt.close()
