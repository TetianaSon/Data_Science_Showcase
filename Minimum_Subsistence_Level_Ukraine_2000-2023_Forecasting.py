import math as mt
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs


def parsing_site_minfin(URL):
    """
    The function that parses data and save it into the file
    :param URL: class 'str'
    :return: class 'list'
    """

    r = requests.get(URL)
    print(r.status_code)
    soup = bs(r.text, "html.parser")
    tables = [
        [
            [td.get_text(strip=True) for td in tr.find_all('td')]
            for tr in table.find_all('tr')
        ]
        for table in soup.find_all('table')
    ]
    table_data = tables[0]
    return table_data


def file_parsing(URL, File_name, Data_name):
    """
    The function of receiving real data from a file
    :param URL: class 'str'
    :param File_name: class 'str'
    :param Data_name: class 'str'
    :return: class 'np.ndarray'
    """

    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        print(values)
    S_real = np.zeros((len(values)))
    for i in range(len(values)):
        S_real[i] = values[i]
    print('Data source: ', URL)
    return S_real


def randomAM(n, iter):
    """
    The function that describes the Uniform Distribution of numbers of anomalous values within the sample
    :param n: class 'int'
    :param iter: class 'int'
    :return: class 'np.ndarray'
    """
    SAV = np.zeros((nAV))
    S = np.zeros((n))
    for i in range(n):
        S[i] = np.random.randint(0, iter)
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    for i in range(nAV):
        SAV[i] = mt.ceil(np.random.randint(1, iter))  # the uniform distribution of the numbers of AV
    print('the numbers of AV: SAV=', SAV)
    print('--------- Statistical Characteristics of UNIFORM Distribution of Random Variables ---------')
    print('mean of Random Variables=', mS)
    print('variance of Random Variables =', dS)
    print('Standard Deviation of Random Variables=', scvS)
    print('-----------------------------------------------------------------------')
    # histogram of the distribution of Random Variables
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return SAV


# ----------------------- Exponential Distribution of a random variable --------------------------
def randoExponential(alfa, iter):
    """
    The function that describes the Exponential Distribution of a random variable
    :param alfa: class 'int'
    :param iter: class 'int'
    :return: class 'np.ndarray'
    """

    S = np.random.exponential(alfa,
                              iter)  # Exponential distribution with a sample of size 'iter' and a parameter 'alfa'.
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('--------- Statistical Characteristics of Exponential Distribution of Random Variables ---------')
    print('mean of Random Variables = ', mS)
    print('variance of Random Variables = ', dS)
    print('Standard Deviation of Random Variables = ', scvS)
    print('----------------------------------------------------------------------------')
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return S


# ------------------- The ideal trend model (Linear Distribution) ---------------------
def Model_linear(iter, a, b):
    """
    The function that describes the ideal trend model (Linear Distribution)
    :param iter: class 'int'
    :param a: class 'float'
    :param b: class 'float'
    :return: class 'np.ndarray'
    """
    S0 = np.zeros(iter)
    for i in range(iter):
        S0[i] = (a * i + b)  # a linear model of a real process
    return S0


# ------------------- The ideal trend model (Quadratic Distribution)  ------------------
def Model_quadratic(n):
    S0 = np.zeros((n))
    for i in range(n):
        S0[i] = (0.45 * i * i + 12 * i + 250)  # a quadratic model of a real process
    return S0


# ---------------- Modeling data with exponential noise------------
def Model_EXP(SN, S0N, n):
    """
    The function that describes the ideal trend model (Linear Distribution) with Exponential noise
    :param SN: class 'np.ndarray'
    :param S0N: class 'np.ndarray'
    :param n: class 'int'
    :return: class 'np.ndarray'
    """
    SV = np.zeros((n))
    for i in range(n):
        SV[i] = S0N[i] + SN[i]
    return SV


# ----- Modeling data with exponential noise + Anomalous Values (AV)
def Model_EXP_AV(S0, SV, nAV, Q_AV):
    SV_AV = SV
    SSAV = np.random.exponential(Q_AV * alfa, nAV)
    for i in range(nAV):
        k = int(SAV[i])
        SV_AV[k] = S0[k] + SSAV[i]
    return SV_AV


# ----- R-squared (R2) score for model evaluation ---------------------
def r2_score(SL, Yout, Text):
    iter = len(SL)
    numerator = 0
    denominator_1 = 0
    for i in range(iter):
        numerator = numerator + (SL[i] - Yout[i]) ** 2
        denominator_1 = denominator_1 + SL[i]
    denominator_2 = 0
    for i in range(iter):
        denominator_2 = denominator_2 + (SL[i] - (denominator_1 / iter)) ** 2
    r2_score_our = 1 - (numerator / denominator_2)
    print('------------', Text, '-------------')
    print('Number of sample elements =', iter)
    print('Coefficient of Determination (approximation probability) =', r2_score_our)

    return r2_score_our


# ------------------- Global linear deviation ----------
def delta_score(SL, Yout, Text):
    iter = len(SL)
    Delta = 0
    for i in range(iter):
        Delta = Delta + abs(SL[i] - Yout[i])
    Delta = Delta / (iter - 1)
    print('-------------', Text, '-------------')
    print('Global linear deviation', Delta)
    return Delta


def comparison_of_indicators(r2_1, r2_2, delta_1, delta_2):
    if (r2_1 > r2_2) and (delta_1 < delta_2):
        print('According to the quality indicators, a Linear model was chosen')
    elif (r2_1 < r2_2) and (delta_1 > delta_2):
        print('According to the quality indicators, a Quadric model was chosen')
    else:
        print('The calculated quality metrics do not ensure a definitive choice of the model')

    return


# ----- Statistical characteristics of the sample  --------------------------
def stat_characteristics_in(SL, Text):
    """
    The function that calculates the statistical characteristics of the sample
    taking into account the trend according to the initial data;
    :param SL: class 'np.ndarray'
    :param Text: class 'str'
    :return: no-return
    """
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print('------------', Text, '-------------')
    print('Number of sample elements = ', iter)
    print('mean of Random Variables = ', mS)
    print('variance of Random Variables = ', dS)
    print('Standard Deviation of Random Variables = ', scvS)
    print('-----------------------------------------------------')
    return


# ----------- Statistical characteristics of the trend  ---------------------
def stat_characteristics_out(SL_in, SL, Text):
    """
    The function that calculates the statistical characteristics of the trend
    :param SL_in: class 'np.ndarray'
    :param SL: class 'np.ndarray'
    :param Text: class 'str'
    :return: no-return
    """
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))
    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    # Global linear deviation
    Delta = 0
    for i in range(iter):
        Delta = Delta + abs(SL_in[i] - Yout[i, 0])
    Delta_average_Out = Delta / (iter + 1)
    print('------------', Text, '-------------')
    print('Number of sample elements = ', iter)
    print('mean of Random Variables = ', mS)
    print('variance of Random Variables = ', dS)
    print('Standard Deviation of Random Variables = ', scvS)
    print('Dynamic model discrepancy =', Delta_average_Out)
    print('-----------------------------------------------------')
    return


# ----- Statistical characteristics of the extrapolation  --------------------------------
def stat_characteristics_extrapol(koef, SL, Text):
    """
    The function that calculates the statistical characteristics of the extrapolation
    :param koef: class 'int'
    :param SL: class 'np.ndarray'
    :param Text: class 'str'
    :return: no-return
    """
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))
    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    #  Confidence interval of forecasted values based on the standard deviation
    scvS_extrapol = scvS * koef
    print('------------', Text, '-------------')
    print('Number of sample elements = ', iter)
    print('mean of Random Variables = ', mS)
    print('variance of Random Variables = ', dS)
    print('Standard Deviation of Random Variables = ', scvS)
    print('Confidence interval of forecasted values based on the standard deviation = ', scvS_extrapol)
    print('-----------------------------------------------------')
    return


# ------------- The Least Squares Method to determine statistical characteristics -------------
def MNK_Stat_characteristics(S0):
    """
    The function of applying the least squares method to determine statistical characteristics
    :param S0: class 'np.ndarray'
    :return: class 'np.ndarray'
    """
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # formation of the structure of input matrices
        Yin[i, 0] = float(S0[i])  # formation of the input data matrix
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout


# --------------- Graphs of trend and measurements with exponential noise  ---------------------------
def Plot_AV(S0_L, SV_L, Text):
    """
    The function that builds graphs of trend and measurements with exponential noise
    :param S0_L: class 'np.ndarray'
    :param SV_L: class 'np.ndarray'
    :param Text: class 'str'
    :return: no-return
    """

    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.show()
    return


# ------------------------------ The Least Squares Method for smoothing -------------------------------------
def MNK(S0):
    """
    The function of applying the least squares method for smoothing
    :param S0: class 'np.ndarray'
    :return: class 'np.ndarray'
    """
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # formation of the structure of input matrices
        Yin[i, 0] = float(S0[i])  # formation of the input data matrix
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    print('Регресійна модель:')
    print('y(t) = ', C[0, 0], ' + ', C[1, 0], ' * t', ' + ', C[2, 0], ' * t^2')
    return Yout


# ----------------- The Least Squares Method for detection and cleaning of anomalous values ---------------------
def MNK_AV_Detect(S0):
    """
    The function of applying the least squares method for detection and cleaning of anomalous values
    :param S0: class 'np.ndarray'
    :return: class 'np.float'
    """
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # formation of the structure of input matrices
        Yin[i, 0] = float(S0[i])  # formation of the input data matrix
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    return C[1, 0]


# --------------------------- The Least Squares Method for extrapolation -------------------------------
def MNK_Extrapol(S0, koef):
    """
    The function of applying the least squares method for extrapolation
    :param S0: class 'np.ndarray'
    :param koef: class 'int'
    :return: class 'np.ndarray'
    """
    iter = len(S0)
    Yout_Extrapol = np.zeros((iter + koef, 1))
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # formation of the structure of input matrices
        Yin[i, 0] = float(S0[i])  # formation of the input data matrix
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    print('Regression model:')
    print('y(t) = ', C[0, 0], ' + ', C[1, 0], ' * t', ' + ', C[2, 0], ' * t^2')
    for i in range(iter + koef):
        Yout_Extrapol[i, 0] = C[0, 0] + C[1, 0] * i + (C[2, 0] * i * i)  # Polynomial curve of the least squares method - forecasting
    return Yout_Extrapol


# --------------------- Detecting anomalous values according to the least squares method ------------------------------
def Sliding_Window_AV_Detect_MNK(S0, Q, n_Wind):
    """
    The function that detects anomalous values according to the least squares method
    :param S0: class 'np.ndarray'
    :param Q: class 'int'
    :param n_Wind: class 'int'
    :return: class 'np.ndarray'
    """
    # ---- Cycle parameters ----
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    # -------- Reference  ---------
    Speed_standart = MNK_AV_Detect(SV_AV)
    Yout_S0 = MNK(SV_AV)
    # ---- Sliding window ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = (j + i)
            S0_Wind[i] = S0[l]
        # -- Statistical characteristics of a sliding window --
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
        # --- Outlier detection and replacement of anomalous values --
        Speed_standart_1 = abs(Speed_standart * mt.sqrt(iter))
        Speed_1 = abs(Q * Speed_standart * mt.sqrt(n_Wind) * scvS)
        if Speed_1 > Speed_standart_1:
            # Outlier detector
            S0[l] = Yout_S0[l, 0]
    return S0


# ------------------- Detecting anomalous values according to the sliding window method ------------------------------
def Sliding_Window_AV_Detect_sliding_wind(S0, n_Wind):
    """
    The function that detects anomalous values according to the sliding window method
    :param S0: class 'np.ndarray'
    :param n_Wind: class 'int'
    :return: class 'np.ndarray'
    """
    # ---- Cycle parameters ----
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    Midi = np.zeros((iter))
    # ---- Sliding window ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = (j + i)
            S0_Wind[i] = S0[l]
        # -- Statistical characteristics of a sliding window --
        Midi[l] = np.median(S0_Wind)
    # ---- Clear sample  -----
    S0_Midi = np.zeros((iter))
    for j in range(iter):
        S0_Midi[j] = Midi[j]
    for j in range(n_Wind):
        S0_Midi[j] = S0[j]
    return S0_Midi


# -------------------------------- Main Call Block ------------------------------------

if __name__ == '__main__':
    # ------------------------ Data parsing from the Minfin website----------------------------------
    URL = "https://index.minfin.com.ua/ua/labour/wagemin/"
    parsing_site_minfin(URL)

    columns_names = ['Period', 'Total', 'Children < 6 years old', 'Children 6 - 18 years old',
                     'Persons capable of working', 'Persons who have lost working capacity']

    df = pd.DataFrame(data=parsing_site_minfin(URL), columns=columns_names)
    df = df.dropna()
    df = df.iloc[::-1]
    df.to_excel("Minfin_LivingWage.xlsx")

    # ------------------------------ Constant segment ----------------------------------------
    n = 70  # number of realizations of Random Values - sample size
    iter = int(n)
    Q_AV = 6  # preference coefficient of anomalous values
    nAVv = 10
    nAV = int((iter * nAVv) / 100)  # number of anomalous values in percentages and absolute units
    alfa = 60  # parameter of the exponential distribution of Random Values
    a = 30
    b = 5

    # ------------------------------ Data Segment -------------------------------------------
    # ------------ Function Calls for Model: Trend, Anomalous, and Exponential Noise  ------
    # -------------------------- Model of the ideal trend (Linear distribution) -------------------
    S01 = Model_linear(iter, a, b)
    SAV = randomAM(n, iter)  # Model of uniform numbers anomalous values
    S = randoExponential(alfa, iter)  # Model of exponential errors

    # ----------------------------- Exponential errors ------------------------------------
    SV1 = Model_EXP(S, S01, n)  # Model of trend + exponential errors
    Plot_AV(S01, SV1, 'Linear model + exponential noise')
    stat_characteristics_in(SV1, 'Sample + exponential noise')

    # ----------------------------- Anomalous errors ------------------------------------
    SV_AV1 = Model_EXP_AV(S01, SV1, nAV, Q_AV)  # model of trend + exponential errors + anomalous values
    Plot_AV(S01, SV_AV1, 'Linear model + exponential noise + anomalous values')
    stat_characteristics_in(SV_AV1, 'Sample + anomalous values')

    # -------------------------- Model of an ideal trend (Quadratic distribution) -------------------
    S02 = Model_quadratic(n)
    SAV = randomAM(n, iter)  # Model of uniform numbers anomalous values
    S2 = randoExponential(alfa, iter)  # Model of exponential errors

    # ----------------------------- Exponential errors ------------------------------------
    SV2 = Model_EXP(S2, S02, n)  # Model of trend + exponential errors
    Plot_AV(S02, SV2, 'Quadratic model + exponential noise')
    stat_characteristics_in(SV2, 'Sample + exponential noise')

    # ----------------------------- Anomalous errors ------------------------------------
    SV_AV2 = Model_EXP_AV(S02, SV2, nAV, Q_AV)  # model of trend + exponential errors + anomalous values
    Plot_AV(S02, SV_AV2, 'Quadratic model + exponential noise + anomalous values')
    stat_characteristics_in(SV_AV2, 'Sample + anomalous values')

    # -------------------------------- Real Data -------------------------------------------
    SV_AV_R = file_parsing('https://index.minfin.com.ua/ua/labour/wagemin/', 'Minfin_LivingWage.xlsx', 'Total')
    SV_AV = SV_AV_R
    Plot_AV(SV_AV_R, SV_AV_R, 'Minimum subsistence level in Ukraine from 2000 to 2023 (UAH)')
    stat_characteristics_in(SV_AV_R, 'Minimum subsistence level in Ukraine from 2000 to 2023 (UAH)')

    # ------------------------- Outlier Removal with Least Squares Method --------------------------
    print('Sample cleaned from anomalous values using MNK method')
    n_Wind = 4  # size of the sliding window for anomalous values detection
    Q_MNK = 6  # detection coefficient
    S_AV_Detect_MNK = Sliding_Window_AV_Detect_MNK(SV_AV, Q_MNK, n_Wind)
    stat_characteristics_in(S_AV_Detect_MNK, 'Sample cleaned from anomalous values using MNK algorithm')
    Yout_SV_AV_Detect_MNK = MNK(S_AV_Detect_MNK)
    stat_characteristics_out(SV_AV, Yout_SV_AV_Detect_MNK, 'MNK Sample cleaned from anomalous values using MNK algorithm')
    Plot_AV(SV_AV_R, S_AV_Detect_MNK, 'Sample cleaned from anomalous values using MNK algorithm')

    # ----------------- Determination of Quality Metrics and Model Optimization ----------------
    # ---------------------- Determination of Quality Metrics ------------------------------------
    r2_linear = r2_score(SV_AV_R, SV_AV1, 'R-squared for linear model')
    print(r2_linear)
    r2_quadric = r2_score(SV_AV_R, SV_AV2, 'R-squared for quadratic model')
    print(r2_quadric)
    delta_linear = delta_score(SV_AV_R, SV_AV1, 'Global linear deviation for linear model')
    print(delta_linear)
    delta_quadric = delta_score(SV_AV_R, SV_AV2, 'Global linear deviation for quadratic model')
    print(delta_quadric)
    comparison_of_indicators(r2_linear, r2_quadric, delta_linear, delta_quadric)
    # ------------------------------- Statistical Learning -----------------------------------
    # ---------------------------- Least Squares Smoothing -----------------------------------
    print('MNK smoothed sample cleaned from anomalous values using sliding window algorithm')
    # -------------------------- Outlier Removal with Sliding Window -------------------------
    n_Wind = 5  # size of the sliding window for anomalous values detection
    S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
    stat_characteristics_in(S_AV_Detect_sliding_wind, 'Sample cleaned from anomalous values using sliding window algorithm')
    Yout_SV_AV_Detect_sliding_wind = MNK(S_AV_Detect_sliding_wind)
    stat_characteristics_out(SV_AV, Yout_SV_AV_Detect_sliding_wind,
                             'MNK smoothed, sample cleaned from anomalous values using sliding window algorithm')
    # --------------- Model Quality Evaluation and Visualization -------------------------------
    r2_score(S_AV_Detect_sliding_wind, Yout_SV_AV_Detect_sliding_wind, 'MNK_Smoothing_Model')
    Plot_AV(Yout_SV_AV_Detect_sliding_wind, S_AV_Detect_sliding_wind,
            'MNK Sample cleaned from anomalous values using sliding window algorithm')

    # -------------------------------------- MNK Forecasting --------------------------------
    print('MNK Forecasting')
    # ------------------------- Outlier Removal with Sliding Window -------------------------
    n_Wind = 5  # size of the sliding window for anomalous values detection
    koef_Extrapol = 0.5  # forecasting coefficient: the ratio of the observation interval to the forecasting interval
    koef = mt.ceil(n * koef_Extrapol)  # forecasting interval based on the number of measurements in the sample
    S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
    stat_characteristics_in(S_AV_Detect_sliding_wind, 'Sample cleaned from anomalous values using sliding window algorithm')
    Yout_SV_AV_Detect_sliding_wind = MNK_Extrapol(S_AV_Detect_sliding_wind, koef)
    stat_characteristics_extrapol(koef, Yout_SV_AV_Detect_sliding_wind,
                                  'MNK Forecasting, sample cleaned from anomalous values using sliding window algorithm')
    Plot_AV(Yout_SV_AV_Detect_sliding_wind, S_AV_Detect_sliding_wind,
            'MNK Forecasting: Sample cleaned from anomalous values using sliding window algorithm')

    '''
    Analysis of the Obtained Results - Verification of Mathematical Models and Calculation Results.
    
    Two input data models are considered:
    1. Linear model.
    Statistical characteristics:
    Random variable distribution - exponential.
        n = 70          # number of random variable realizations - sample size.  
        alfa = 60       # parameter of the exponential distribution of the random variable
        a = 30
        b = 5           # parameters of the linear distribution of the random variable
        
    2. Quadratic model. 
    Statistical characteristics:
    Random variable distribution - exponential.
        n = 70          # number of random variable realizations - sample size
        alfa = 60       # parameter of the exponential distribution of the random variable.
        y(t) = 250 + 12*t + 0.45*t^2    # quadratic model

    3. Determined characteristics of the input sample (linear model):
    The temporal trend of data with a linear distribution is confirmed by the graph;
    Statistical characteristics:
        The distribution of random variables is exponential, confirmed by the histogram;
        -----------------------------------------------------------------------
        ------- Statistical Characteristics of the EXPONENTIAL Measurement Error -----
        Mean of the random variable = 33.65526720079783
        Variance of the random variable = 3484.7288158855126
        Standard Deviation of the random variable = 59.03159167670742
        -----------------------------------------------------------------------
        ------------ Sample with Exponential Noise -------------
        Number of elements in the sample = 70
        Mean of the random variable = -25.321726478645246
        Variance of the random variable = 3475.205290947461
        Standard Deviation of the random variable = 58.95087184213191
        -----------------------------------------------------------------------
        ------------ Вибірка з АВ -------------
        Number of elements in the sample = 70
        Mean of the random variable = -42.29189694270366
        Variance of the random variable = 10261.602620778654
        Standard Deviation of the random variable = 101.29956870973663
    4. Determined characteristics of the input sample (quadratic model):
        The temporal trend of data with a quadratic distribution is confirmed by the graph;
        Statistical characteristics:
        The distribution of random variables is exponential, confirmed by the histogram;
        -----------------------------------------------------------------------
        ------- Statistical Characteristics of the EXPONENTIAL Measurement Error -----
        Mean of the random variable = 31.952583565547762
        Variance of the random variable = 7851.9840105961075
        Standard Deviation of the random variable = 88.61142144552308
        -----------------------------------------------------------------------
        ------------ Sample with Exponential Noise -------------
        Number of elements in the sample = 70
        Mean of the random variable = -33.52749883172663
        Variance of the random variable = 7791.724732634365
        Standard Deviation of the random variable = 88.27074675471124
        -----------------------------------------------------------------------
        ------------ Sample with anomalous values -------------
        Number of elements in the sample = 70
        Mean of the random variable = -52.23465766572372
        Variance of the random variable = 23610.23002628366
        Standard Deviation of the random variable = 153.65620724944262
    5. Real Data
        ------------ Minimum Subsistence Level in Ukraine from 2000 to 2023 (UAH) -------------
        Number of elements in the sample = 57
        Mean of the random variable = 6.548463502413142
        Variance of the random variable = 2509.620362122619
        Standard Deviation of the random variable = 50.09611124750722
        
    6. Data Cleaning from Anomalous Measurements
        The sample is cleaned from anomalous values using the MNK method
        Regression model:
        y(t) =  354.58857547139553  +  13.624690345491086  * t  +  0.47493920764399633  * t^2
        ------------ The sample is cleaned from anomalous values using the MNK method -------------
        Number of elements in the sample = 57
        Mean of the random variable = -0.5944111462104047
        Variance of the random variable = 135.9131084960953
        Standard Deviation of the random variable = 11.658177751951431
        Regression model (after cleaning from anomalous values) :
        y(t) =  328.9183261960843  +  15.405553628772463  * t  +  0.44893834123715926  * t^2
        ---------- Sample cleaned from anomalous values using the MNK method -------------
        Number of elements in the sample = 57
        Mean of the random variable = 3.637978807091713e-12
        Variance of the random variable = 2.3487691177897308e-23
        Standard Deviation of the random variable = 4.84641013306729e-12
        Dynamic model error = 6.890470115006201
    7. Determination of Quality Indicators and Model Optimization
        ------------ R-squared for the linear model -------------
        Number of elements in the sample = 57
        Coefficient of determination (approximation probability) = 0.6977215473825875
        ------------ R-squared for the quadratic model -------------
        Number of elements in the sample = 57
        Coefficient of determination (approximation probability) = 0.9241182831138652
        ------------- Global linear deviation of the linear model -------------
        Global linear deviation of the linear model estimate: 314.461828556180
        ------------- Global linear deviation of the quadratic model -------------
        Global linear deviation of the quadratic model estimate: 144.99871360500003
        
        Based on the quality indicators, the quadratic model is chosen.
        
    8. Statistical Learning   
        ------------ The sample is cleaned from anomalous values using the sliding window algorithm -------------
        Number of elements in the sample = 57
        Mean of the random variable = -0.44380766743756794
        Variance of the random variable = 110.11798259957776
        Standard Deviation of the random variable = 10.493711574060807
        -----------------------------------------------------
        Regression model:
        y(t) =  321.94738725660324  +  12.281891025796885  * t  +  0.4664297621544661  * t^2
        --------- Sample characteristics after cleaning from anomalous values --------------------- 
        ---------- using the sliding window algorithm (after cleaning) ----------------------------
        Number of elements in the sample = 57
        Mean of the random variable = 2.7284841053187847e-12
        Variance of the random variable = 2.3691214381715462e-23
        Standard Deviation of the random variable = 4.867362158470999e-12
        Dynamic model error = 77.50404726467002
        -----------------------------------------------------
        ------------ MNK Smoothing Model -------------
        Number of elements in the sample = 57
        Coefficient of determination (probability of approximation) = [0.99973273]
    9. MNK Forecasting
        -------- Sample characteristics after cleaning from anomalous values using the sliding window algorithm  -------
        Number of elements in the sample = 57
        Mean of the random variable = -0.44380766743756794
        Variance of the random variable = 110.11798259957776
        Standard Deviation of the random variable = 10.493711574060807
        -----------------------------------------------------
        Regression model:
        y(t) =  321.94738725660324  +  12.281891025796885  * t  +  0.4664297621544661  * t^2
        
        ----------------------------- MNK Forecasting  ------------------------------------------
        ------- Sample characteristics after cleaning from anomalous values using the sliding window algorithm ---------
        Mean of the random variable = -0.44380766743756794
        Variance of the random variable = 110.11798259957776
        Standard Deviation of the random variable = 10.493711574060807
        ------------ MNK FORECASTING, sample cleaned from anomalous values using sliding_wind algorithm -------------
        Number of elements in the sample = 92
        Mean of the random variable = -6.480149750132114e-12
        Variance of the random variable = 1.2491782785182814e-22
        Standard Deviation of the random variable = 1.1176664433176302e-11
        Confidence interval for forecasted values using standard deviation = 3.9118325516117057e-10

    3. Conclusion
    The correspondence between the specified and calculated numerical characteristics of the statistical sample 
    demonstrates the adequacy of the calculations.
    Out of the two models (linear and quadratic), the quadratic model was chosen based on the 
    calculated quality indicators.
    Data cleaning and statistical learning were performed using the method of least squares (MNK)
    Forecasting of the parameters of the studied process was conducted.
    '''
