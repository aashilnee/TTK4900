'''
def iserror(func, *args, **kw):
    try:
        func(*args, **kw)
        return False
    except Exception:
        return True


#------------------ FINNE VERDIENE TIL LOSS, legges til i loss.txt
import re
key = "loss:"
with open('trainlog_without_data_augmentation_1054_epochs.txt') as in_file:
    #number = 0
    for line in in_file:
        match = re.search('loss:', line)
        if match:
            if iserror(float, line[33:39]) == False:
                print(float(line[33:39]))
                f = open("loss.txt", "a+")
                f.write(line[33:39])
            #f.write(' ' + str(number))
                f.write("\n")
            elif iserror(float, line[34:40]) == False:
                print(float(line[34:40]))
                f = open("loss.txt", "a+")
                f.write(line[34:40])
                # f.write(' ' + str(number))
                f.write("\n")

            else:
                print("FEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEIL")

            #number +=1

'''

'''
#------------------ FINNE GJENNOMSNITT, resultat per epoke, lagt i average.txt
with open('loss.txt') as fh:
    sum = 0
    number = 0
    x = 0
    for line in fh:

        number += 1
        sum += float(line.split()[0])
        if number % 1000 == 0:  # antall bilder
            x += 1
            average = sum / number
            f = open("average.txt", "a+")
            f.write(str(x)+ ' ')
            f.write(str(average))

            f.write("\n")
            number = 0
            sum = 0
            x=x

'''


#------------------ PLOT 
import matplotlib.pyplot as plt
with open('average.txt', 'r') as f:
    lines = f.readlines()
    x = [float(line.split()[0]) for line in lines]
    y = [float(line.split()[1]) for line in lines]
plt.title("Total loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(x ,y)
plt.show()

