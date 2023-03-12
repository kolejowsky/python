import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def DFT_fun(x):

    N = len(x)

    DFT = 0 #zmienna przechowujaca wartosci harmonicznych
    DFT_list = [] #lista przechowujaca wynik obliczen DFT
    for n in range(0, N):
        for i in range(0, N):
            tempDFT = x[i] * np.exp((-2j * np.pi * n * i)/N) #liczenie pojedynczego elementu sumy DFT
            DFT += tempDFT

        DFT_list.append(DFT)

        DFT = 0

    return DFT_list

def FFT_fun(x):
    N = len(x)

    if N == 1:
        return x

    parz = x[0::2]  # co drugi element tablicy liczony od 0 => parzyste
    nparz = x[1::2]  # co drugi element tablicy liczony od 1 => nieparzyste

    #wywolanie funkcji rekurencyjnie
    parz_FFT = FFT_fun(parz)
    nparz_FFT = FFT_fun(nparz)


    FFT_list = [0] * N #lista o rozmiarze N

    for k in range(N//2):
        temp1 = np.exp(-2j*np.pi*k/N) * nparz_FFT[k]
        FFT1 = parz_FFT[k] + temp1 #liczenie harmonicznych o parzystych indeksach
        FFT_list[k] = FFT1 #przypisanie do odpowiedniego miejsca w tablicy wynikow

        temp2 = np.exp(-2j*np.pi*k/N) * nparz_FFT[k]
        FFT2 = parz_FFT[k] - temp2 #liczenie harmonicznych o nieparzystych indeksach
        FFT_list[k + N//2] = FFT2 #przypisanie do odpowiedniego miejsca w tablicy wynikow

    return FFT_list

def IDFT_fun(DFT_list):
    N = len(DFT_list)
    IDFT = 0 #zmienna przechowujaca wynik obliczen IDFT
    IDFT_list = [] #lista przechowujaca wynik IDFT
    for n in range(0, N):
        for k in range(0, N):
            tempIDFT = (1/N)*DFT_list[k]*np.exp((2j * np.pi * n * k)/N) #liczenie pojedynczego elementu sumy IDFT
            IDFT += tempIDFT

        IDFT_list.append(IDFT)
        IDFT = 0

    return IDFT_list

def IFFT_fun(DFT_list):
    IFFT_list = np.fft.ifft(DFT_list) #obliczanie za pomoca gotowej funkcji
    return IFFT_list

def FFT_2D(x):
    return np.fft.fft2(x)

def DFT_2D(x):
    x = np.array(x)
    N = x.shape[0]
    M = x.shape[1]
    X = np.zeros((N, M), dtype=complex)
    for k1 in range(N):
        for k2 in range(M):
            for n1 in range(N):
                for n2 in range(M):
                    X[k1, k2] += x[n1, n2] * np.exp(-2j * np.pi * (n1 * k1 / N + n2 * k2 / M))
    return X.tolist()

def plot_signal1D(x):

    #plt.plot(x)
    plt.plot(range(len(x)), x, '--')
    plt.scatter(range(len(x)), x)
    plt.xlabel('Numer pomiaru')
    plt.ylabel('Amplituda')
    plt.title('Sygnał')
    plt.show()

def plot_x(x):
    plt.stem(x)
    #plt.scatter(range(len(x)), x)
    plt.xlabel('n')
    plt.ylabel('Amplituda')
    plt.title('Widmo wspołczynników po zastosowaniu DFT')
    plt.show()

def plot_signal2D(x):
    plt.imshow(x, cmap=cm.gray)
    plt.xlabel('Numer pomiaru X')
    plt.ylabel('Numer pomiaru Y')
    plt.title('Sygnał 2D')
    plt.show()

def plot2D_x(x):
    magnitudes = np.abs(x)

    plt.pcolor(magnitudes, cmap=cm.gray)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Widmo wspołczynników po zastosowaniu DFT 2D')
    plt.show()

def print_result(x):

    N = len(x)

    print("Dane")
    print("\n")

    for i in range(len(x)):
        print(i, end=" ")
        print("{:.3f}".format(x[i]))

    #plot_signal1D(x)

    print("\n")
    print("DFT")
    print("\n")

    DFT_list = DFT_fun(x)

    for i in range(len(DFT_list)):
        print(i, end=" ")
        print("{:.3f}".format(DFT_list[i]))

    #plot_x(DFT_list)

    print("\n")
    print("FFT")
    print("\n")

    FFT_list = FFT_fun(x)

    for i in range(len(FFT_list)):
        print(i, end=" ")
        print("{:.3f}".format(FFT_list[i]))

    print("\n")
    print("IDFT")
    print("\n")

    IDFT_list = IDFT_fun(DFT_list)

    for i in range(len(DFT_list)):
        print(i, end=" ")
        print("{:.3f}".format(IDFT_list[i]))

    print("\n")
    print("IFFT")
    print("\n")

    IFFT_list = IFFT_fun(DFT_list)

    for i in range(len(IFFT_list)):
        print(i, end=" ")
        print("{:.3f}".format(IFFT_list[i]))

def print_2d(x):

    #plot_signal2D(x)


    print("DFT 2D")

    DFT2D_list = DFT_2D(x)
    for i in range(len(DFT2D_list)):
        for j in range(len(DFT2D_list[i])):
            print("{:.3f}".format(DFT2D_list[i][j]), end="")
        print("")

    print("\n")



    print("FFT 2D")
    
    FFT2D_list = FFT_2D(x)
    for i in range(len(FFT2D_list)):
        for j in range(len(FFT2D_list[i])):
            print("{:.3f}".format(FFT2D_list[i][j]), end="")
        print("")

    #plot2D_x(FFT2D_list)


def main():

    x = [1, 2, 1, 2]

    #otwieranie pliku

    filename = sys.argv[1]

    with open(filename) as f:
    #with open("dane2_05.in") as f:
        dimension = int(f.readline())

        measurements = []

        if (dimension == 1):
            num_measurements = int(f.readline())

            for i in range(num_measurements):
                measurements.append(float(f.readline()))

        elif(dimension == 2):

            num_measurements = f.readline().strip().split(" ")
            # pobierz wymiary
            wym_x, wym_y = int(num_measurements[0]), int(num_measurements[1])

            for line in f:
                line = line.replace("  ", " ")
                measurements.append(line.strip().split(" "))

            for i, row in enumerate(measurements):
               for j, val in enumerate(row):
                    measurements[i][j] = float(val)


    x = measurements
    
    #for i in range(len(measurements)):
        #print(measurements[i])

    #print_result(x)

    if(dimension == 1):
        print_result(x)
    elif(dimension == 2):
        print_2d(x)
        #for i in range(len(x)):
            #print(x[i])


    #print(IDFT_list[0::2])


main()