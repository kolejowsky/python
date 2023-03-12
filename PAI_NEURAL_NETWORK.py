import numpy as np
import random
import matplotlib.pyplot as plt

input_num = 35

#reprezentacja litery A
CorrectA = [1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1]

#reprezentacja litery C
CorrectC = [1, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 1]

correct_letters = [CorrectA, CorrectC]
target = [0, 1]

#Set przygotowany do ucznia
learnSet = [[correct_letters[0], target[0]], [correct_letters[1], target[1]]]

#zaszumione litery A (zmienione zostalo 5 pikseli)
inCorrectA1 = [1, 1, 0, 1, 1,
               1, 1, 0, 0, 1,
               1, 0, 0, 0, 0,
               1, 1, 1, 0, 1,
               1, 0, 0, 0, 1,
               1, 0, 1, 0, 1,
               1, 0, 0, 0, 1]

inCorrectA2 = [1, 1, 1, 1, 1,
               1, 0, 0, 0, 1,
               0, 1, 0, 0, 1,
               1, 1, 1, 1, 1,
               1, 0, 0, 0, 0,
               1, 1, 0, 0, 1,
               1, 0, 0, 1, 1]

inCorrectA3 = [0, 1, 1, 1, 1,
               1, 0, 0, 1, 1,
               1, 0, 0, 0, 1,
               0, 1, 1, 1, 1,
               1, 0, 0, 0, 1,
               1, 0, 0, 1, 1,
               0, 0, 0, 0, 1]

inCorrectA4 = [1, 1, 1, 1, 0,
               1, 1, 0, 0, 1,
               1, 0, 0, 0, 0,
               0, 1, 1, 1, 1,
               1, 1, 0, 0, 1,
               1, 0, 0, 0, 1,
               1, 0, 0, 0, 1]

inCorrectA5 = [1, 0, 1, 0, 1,
               1, 0, 0, 0, 1,
               1, 0, 1, 0, 1,
               1, 1, 1, 1, 1,
               1, 0, 0, 0, 1,
               1, 0, 0, 0, 1,
               1, 0, 1, 0, 0]

#zaszumione litery C (zmienione 5 pikseli)
inCorrectC1 = [1, 1, 1, 1, 1,
               1, 0, 0, 0, 0,
               1, 0, 1, 0, 0,
               1, 0, 0, 0, 1,
               1, 1, 0, 1, 0,
               1, 0, 0, 0, 0,
               1, 1, 0, 1, 1]

inCorrectC2 = [1, 0, 1, 1, 1,
               1, 0, 0, 0, 0,
               0, 0, 1, 0, 0,
               1, 0, 0, 1, 0,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 1, 0, 1, 1]

inCorrectC3 = [1, 1, 1, 1, 0,
               1, 1, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0,
               0, 0, 0, 1, 0,
               1, 0, 0, 0, 0,
               0, 1, 1, 1, 1]

inCorrectC4 = [1, 1, 1, 1, 1,
               1, 0, 1, 0, 0,
               1, 0, 0, 0, 1,
               1, 0, 0, 1, 0,
               1, 0, 1, 0, 0,
               1, 0, 0, 0, 0,
               1, 1, 1, 0, 1]

inCorrectC5 = [1, 0, 1, 1, 1,
               0, 0, 0, 0, 0,
               1, 0, 0, 1, 0,
               1, 1, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 1,
               1, 1, 1, 1, 1]

incorrect_letters = [inCorrectA1, inCorrectA2, inCorrectA3, inCorrectA4, inCorrectA5,
                     inCorrectC1, inCorrectC2, inCorrectC3, inCorrectC4, inCorrectC5]
test_target = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

#Set przygotowany do testowania
testSet = [[incorrect_letters[0], test_target[0]], [incorrect_letters[1], test_target[1]],
           [incorrect_letters[2], test_target[2]], [incorrect_letters[3], test_target[3]],
           [incorrect_letters[4], test_target[4]], [incorrect_letters[5], test_target[5]],
           [incorrect_letters[6], test_target[6]], [incorrect_letters[7], test_target[7]],
           [incorrect_letters[8], test_target[8]], [incorrect_letters[9], test_target[9]]]

#funkcja zmieniajaca pixel
def change(pixel):
    if(pixel == 0):
        pixel = 1
    elif(pixel == 1):
        pixel = 0
    return pixel

#funkcja pomocnicza losujaca liczbe z przedzialu -1;1
def weight():
    return random.uniform(-1, 1)

#funkcja generujaca wage
def generate_weights(weights, n):
    for i in range(n):
        weights.append(weight())

#wspolczynnik uczenia
learningFactor = 0.5

#tablica wag
weights1 = []

#weights2 = []


'''
neuron1 = []
for i in range(input_num):
    neuron1.append(0)
'''

#zmienna przechowujaca sume iloczynow wag i wartosci inputu
neuron1 = 0

#neuron2 = 0

#funkcja sigmoid
def sig(x):
    result = 1/(1+np.exp(-1*x))
    return result

#pochodna funkcji sigmoid (jako argument podajemy wynik funkcji sig dla danego argumentu)
def sig_prim(y):
    result = (1-y)*y
    return result

#funckja wyliczajaca blad
def error_fun(t, y):
    E = 0.5*((t-y)**2)
    return E

#funkcja wyliczajaca nowe wagi
def new_weight(weight, lF, t, y, x):
    n_weight = weight + lF * (t - y) * sig_prim(y) * x
    #n_weight = weight + lF * (t - y) * x * (1 - x)
    return n_weight

#funkcja pomocnicza wypisujaca w sposob czytelny litery na ekran
def print_letter(letter):
    for i in range(len(letter)):
        if (i % 5 == 0):
            print("")

        if(letter[i] == 1):
            print("#", end=" ")
        else:
            print(" ", end=" ")
    print("")


def main():
    # ------------------------------------------------------------
    #
    #                   Uczenie
    #
    # ------------------------------------------------------------
    errors = [] #tablica przechowujaca bledy po kazdej iteracji

    #testowanie
    #printLetter(learnSet[0])
    #print("")
    #printLetter(learnSet[1])

    # for i in range(len(weights1)):
    #    print(i, " = ", weights1[i])


    #losowanie wag
    generate_weights(weights1, input_num)
    #generate_weights(weights2, input_num)

    #tablica pomocnicza przechowujaca set z literami A i C w losowej kolejnosci
    letters_in = [[], []]


    #Jak odwolac sie do danej czesci Setu

    #tablica z pierwsza litera --> letters_in[0][0]
    #target pierwszej litery --> letters_in[0][1]

    #tablica z druga litera --> letters_in[1][0]
    #target drugiej litery --> letters_in[1][1]

    #parametry
    learningFactor = 0.5 #0.5
    iterations = 800 #800
    show_data = 100 #co ile iteracji wyswietlic informacje
    show_prediction = 50 #co ile iteracji wyswitelic przewidywanie w testowaniu
    current_iteration = 1
    error = 1
    expected_error = 0.001

    print("----------UCZENIE---------- \n")

    #petla z uczeniem
    while(current_iteration <= iterations): #and error > expected_error):
        #losowanie zbioru uczacego
        rand_num = random.randint(0, 1)
        #if (current_iteration%show_data == 0): print(rand_num)
        if (rand_num == 0):
            letters_in[0] = learnSet[0]
            letters_in[1] = learnSet[1]

        else:
            letters_in[0] = learnSet[1]
            letters_in[1] = learnSet[0]


        if(current_iteration % show_data == 0):
            print("iteration ", current_iteration)

        #uczenie na pierwszej literze
        neuron1 = 0 #zmienna przechowujaca sume iloczynow wag i wartosci inputu

        for i in range(input_num):
            neuron1 += weights1[i] * letters_in[0][0][i]

        '''
        neuron2 = 0
        for i in range(input_num):
            neuron2 += neuron1[i] * weights2[i]
        
        '''
        y = sig(neuron1) #wyliczanie wyniku sieci
        er = letters_in[0][1] - y #odchylenie wyniku sieci od targetu
        error = error_fun(letters_in[0][1], y) #wlasciwe wyliczenie bledu

        errors.append(error) #dodanie bledu do tablicy

        #wypisanie wyniku dla pierwszej litery
        if(current_iteration % show_data == 0):
            print("y = ", y)
            #print("t - y = ", er)
            print("error = ", error)

            print("")


        #ustawienie nowych wag
        for i in range(input_num):
            weights1[i] = new_weight(weights1[i], learningFactor, letters_in[0][1], y, letters_in[0][0][i])


        #uczenie na drugiej literze
        neuron1 = 0

        for i in range(input_num):
            neuron1 += weights1[i] * letters_in[1][0][i]

        y = sig(neuron1)
        er = letters_in[1][1] - y
        error = error_fun(letters_in[1][1], y)
        errors.append(error)

        # wypisanie wyniku dla pierwszej litery
        if (current_iteration % show_data == 0):
            print("y = ", y)
            #print("t - y =", er)
            print("error = ", error)
            print("\n\n")

        #ustawienie nowych wag
        # ustawienie nowych wag
        for i in range(input_num):
            weights1[i] = new_weight(weights1[i], learningFactor, letters_in[1][1], y, letters_in[1][0][i])

        #zaktualizowanie wspolczynnika uczenia
        learningFactor *= 0.98

        #inkrementacja zmiennej przechowujacej informacje o iteracji
        current_iteration += 1

        #print(y)

    #wyskres bledu od iteracji
    plt.plot(errors)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()

    #------------------------------------------------------------
    #
    #                   Testowanie
    #
    #------------------------------------------------------------

    print("----------TESTOWANIE---------- \n")

    # tablica z i-1ta litera --> letters_in[i][0]
    # target i-1tej litery --> letters_in[i][1]

    #errors2 = []
    test_iterations = 400 #zmienna przechowujaca informacje o ilosci iteracji testowania
    # ustawiona na polowe iteracji uczenia poniewaz
    # w kazdej iteracji uczenia odbywaly sie 2 wyliczenia dla A i C
    accuracy = 0 #poprawnosc
    wrongPredictions = 0 #ile razy siec sie pomylila
    wrongPredictionsIndex = [] #tablica indeksow liter ktore nie zostaly dobrze rozpoznane
    wrongPredictionsIteration = [] #tablica iteracji w ktorej wystapil blad


    for test_i in range(test_iterations):
        rand_num = random.randint(0, 9) #wybranie losowo elementu ze zbioru do testowania

        neuron1 = 0 #adekwatnie do uczenia

        for i in range(input_num):
             neuron1 += weights1[i] * testSet[rand_num][0][i]

        y = sig(neuron1)
        er = letters_in[1][1] - y
        error = error_fun(letters_in[1][1], y)

        #errors2.append(error)

        if(y <= 0.5):
            prediction = 0
        else:
            prediction = 1

        #sprwadzenie czy przeidywanie jest prawidlowe
        if (prediction == testSet[rand_num][1]):
            isCorrect = True
        else:
            isCorrect = False

        #jesli jest zliczamy do dokladnosci jesli nie do bledow
        if(isCorrect):
            accuracy += 1
        else:
            wrongPredictions += 1
            wrongPredictionsIndex.append(rand_num)  # zapisanie ktorego elementu ze zbioru nie udalo sie odgadnac
            wrongPredictionsIteration.append(test_i)  # zapisanie podczas ktorej iteracji nie udalo sie odgadnac

        #wypisanie przewidywan
        if(test_i % show_prediction == 0): #or isCorrect == False):
            print("Iteration:", test_i)
            print("Result: ", y)
            print("Prediction: ", prediction)
            print("Target: ", testSet[rand_num][1])
            print("Is Correct: ", isCorrect)
            print("")

        #if(isCorrect == False):
        #    print("Wrong prediction on iteration ", test_i)

        #testowanie
        #print(y)
        #print_letter(testSet[rand_num][0])
        #print(testSet[rand_num][1])

    accuracy = (accuracy / test_iterations) * 100 #wyliczenie dokladnosci
    print("Accuracy: ", accuracy, "%")
    print("Wrong predictions: ", wrongPredictions)

    '''
    for i in range(len(wrongPredictionsIteration)):
        print("Iteration: ", wrongPredictionsIteration[i],)
        print("Index: ", wrongPredictionsIndex[i],)
    '''

    #plt.plot(errors2)
    #plt.show()

    '''
    print(rand_num)
    print_letter(letters_in[0][0])
    print(letters_in[0][1])
    print_letter(letters_in[1][0])
    print(letters_in[1][1])
    '''

    return

main()
