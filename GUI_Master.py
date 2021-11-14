import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter import messagebox as ms
import os
import pandas as pd
import csv
import random
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
#######################################################################################################
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

##############################################+=============================================================

root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("CAR Classification Analysis")

##############################################+=============================================================
#####For background Image
image2 =Image.open('car1.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)
background_image=ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image)
background_label.image = background_image

background_label.place(x=100, y=0) #, relwidth=1, relheight=1)
#height=1, width=35,
lbl = tk.Label(root, text="Car Classification according to (PRICE,COMFORT,SAFETY)", font=('times', 25,' bold '),justify=tk.LEFT, wraplength=1300 ,bg="white",fg="indian red")
lbl.place(x=250, y=5)


frame_display = tk.LabelFrame(root, text=" --Car Data-- ", width=1000, height=250, bd=5, font=('times', 10, ' bold '),bg="white",fg="red")
frame_display.grid(row=0, column=0, sticky='s')
frame_display.place(x=200, y=60)

str1="Buying Price-----Maintenance Price-----Doors-----Persons-----Lug_Boot-----Safety"
frame_input = tk.LabelFrame(root, text=str1, width=1000, height=100, bd=5, font=('times', 20, ' bold '),bg="white",fg="red")
frame_input.grid(row=0, column=0, sticky='s')
frame_input.place(x=200, y=320)

frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=180, height=650, bd=5, font=('times', 10, ' bold '),bg="lightblue4")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=5, y=0)

###########################################################################################################


tree = ttk.Treeview(frame_display, columns=('Buying Price','Maintenance Price','Doors','Persons','Lug_Boot','Safety','Values'))
style = ttk.Style()
style.configure('Treeview', rowheight=20)
style.configure("Treeview.Heading", font=(None, 10),rowheight=100)
#style = ttk.Style()
style.configure(".", font=('Helvetica', 10), foreground="blue")
style.configure("Treeview", foreground='red')
style.configure("Treeview.Heading", foreground='red',background='green') #<----


tree["columns"] = ("1", "2", "3", "4", "5", "6", "7")
tree.column("1", width=100)
tree.column("2", width=150)
tree.column("3", width=100)
tree.column("4", width=100)
tree.column("5", width=100)
tree.column("6", width=100)
tree.column("7", width=200)



tree.heading("#0", text='*', anchor='w')
tree.column("#0", anchor="w",width=0)
tree.heading("1", text="Buying Price")
tree.heading("2", text="Maintenance Price")
tree.heading("3", text="Doors")
tree.heading("4", text="Persons")
tree.heading("5", text="Lug Boot")
tree.heading("6", text="Safety")
tree.heading("7", text="Values")

tree.place(x=0, y=0)


V1 =tk.IntVar()
V2 =tk.IntVar()
V3 =tk.IntVar()
V4 =tk.IntVar()
V5 =tk.IntVar()
V6 =tk.IntVar()

def Clear_entry():
    t1.delete(0, END)
    t2.delete(0, END)
    t3.delete(0, END)
    t4.delete(0, END)
    t5.delete(0, END)
    t6.delete(0, END)
    t1.focus_set()

    
t1 =tk.Entry(frame_input, textvar=V1,width=5, font=('', 15),bg='lightblue4',fg='white')
t1.place(x=50, y=7)

t2 =tk.Entry(frame_input, textvar=V2,width=5, font=('', 15),bg='lightblue4',fg='white')
t2.place(x=250, y=7)

t3 =tk.Entry(frame_input, textvar=V3,width=5, font=('', 15),bg='lightblue4',fg='white')
t3.place(x=470, y=7)

t4 =tk.Entry(frame_input, textvar=V4,width=5, font=('', 15),bg='lightblue4',fg='white')
t4.place(x=600, y=7)

t5 =tk.Entry(frame_input, textvar=V5,width=5, font=('', 15),bg='lightblue4',fg='white')
t5.place(x=750, y=7)

t6 =tk.Entry(frame_input, textvar=V6,width=5, font=('', 15),bg='lightblue4',fg='white')
t6.place(x=900, y=7)

t1.focus_set()

################################################################################################################
def load_model():
    import numpy as np
    
    if t2.get():
        with open('clf_NB.pkl', 'rb') as f:
            clf_NB = pickle.load(f)
        
        
        list_val=[float(t1.get()),float(t2.get()),float(t3.get()),float(t4.get()),float(t5.get()),float(t6.get())]
    #    Predict_to_value=np.array([1,15,1,1000])
        Predict_to_value=np.array(list_val)
    
        Predict_to_value=Predict_to_value.reshape(1, -1)
        Predict_get_value=clf_NB.predict(Predict_to_value)
        print(Predict_get_value)
        if Predict_get_value[0]==4:
            msg="unaccepted"
        elif Predict_get_value[0]==3:
            msg="accepted"
        elif Predict_get_value[0]==2:
            msg="good"   
        elif Predict_get_value[0]==1:
            msg="very good"   
            
            
        update_label(msg)
    
    else:
        update_label("Please Enter Car Values to predict!!!!")
        
def update_label(str_T):
    result_label = tk.Label(root, text=str_T, width=50, font=("bold", 20),bg='white',fg='red' )
    result_label.place(x=200, y=570)



#####For Navie Bayes Algorithm


def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
 
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
 
def mean(numbers):
#    print(numbers)
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
 
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries
 
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries
 
def calculateProbability(x, mean, stdev):
    try:
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    except ZeroDivisionError:
        return(0)
    
    
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities
            
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel
 
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions
 
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
####################################################################################################
def cl_NB():

    filename = 'car_int.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    str1='Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet))
    print(str1)
    # prepare model
    summaries = summarizeByClass(trainingSet)

    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)

#    print('Accuracy: {0}%'.format(accuracy))
    
    
    with open('Mclf_NB.pkl', 'wb') as f:
        pickle.dump(summaries, f)

    
    A="Naive Bayes Accuracy: {0}%".format(round(accuracy,2))
    C=direct_NB()

    D = A+'\n'+ C
    
    update_label(D)

def direct_NB():
    
    col_head=['BP','MP','doors','persons','LB','safety','val']
    df = pd.read_csv("car_int.csv",names=col_head) 
    df.sort_values(by=['val'],ascending=True,inplace=True)
    
    Train_col=['BP','MP','doors','persons','LB','safety']
    features = list(df[Train_col])
    target = list(df[['val']])
    
    
    X = df[features] #our features that we will use to predict Y
    Y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=4)

    clf_Naive = GaussianNB()
    clf_Naive.fit(X_train, y_train.values.ravel())
    
    prediction = clf_Naive.predict(X_test)
    
           ## now you can save it to a file
    with open('clf_NB.pkl', 'wb') as f:
        pickle.dump(clf_Naive, f)

    
    C="Naive Bayes Model Saved as <<  clf_NB.pkl  >>"

    return C
################################################################################################################
#buying (buying price): vhigh (4), high (3), med (2), low (1)
# main (maintenance price): vhigh (4), high (3), med (2), low (1)
# doors (number of doors): 2, 3, 4, 5-more (5)
# persons (number of passengers fit in a car): 2, 4, more (6)
# lug_boot (size of luggage capacity): small (1), med (2), big (3)
# safety: low (1), med (2), high (3)
# values: unacc = unaccepted, acc = accepted, good = good, vgood = very good

#'Buying Price','Maintenance Price','Doors','Persons','Lug_Boot','Safety','Values'


def process_data():
    col_head=['BP','MP','doors','persons','LB','safety','val']
    data = pd.read_csv("car.data",names=col_head,header=None) 
    
    #For buying price
    data.BP[data.BP == 'vhigh'] = 4
    data.BP[data.BP == 'high'] = 3
    data.BP[data.BP == 'med'] = 2
    data.BP[data.BP == 'low'] = 1

    #For Maintenance Price
    data.MP[data.MP == 'vhigh'] = 4
    data.MP[data.MP == 'high'] = 3
    data.MP[data.MP == 'med'] = 2
    data.MP[data.MP == 'low'] = 1

    #For doors
    data.doors[data.doors == '5-more'] = 5
    data.doors[data.doors == '5more'] = 5
    #For Persons
    data.persons[data.persons == 'more'] = 6

    #For Lug boot
    data.LB[data.LB == 'small'] = 1
    data.LB[data.LB == 'med'] = 2
    data.LB[data.LB == 'big'] = 3
    
    #For safety 
    data.safety[data.safety == 'low'] = 1
    data.safety[data.safety == 'med'] = 2
    data.safety[data.safety == 'high'] = 3
    
    #For value
    data.val[data.val == 'unacc'] = 4
    data.val[data.val == 'acc'] = 3
    data.val[data.val == 'good'] = 2
    data.val[data.val == 'vgood'] = 1

    data.to_csv("car_int.csv", header=None,index=False)
    tree.delete(*tree.get_children())
    Load_data("car_int.csv")
    
def load_csv():
    Load_data('car.data')
    
    
    
def Load_data(CSV_NAM):

    data = pd.read_csv(CSV_NAM,header=None) 

    for i in range(len(data)):
        tree.insert("", i, text=i+1, values=(data[0][i],
                                           data[1][i],
                                           data[2][i],
                                           data[3][i],
                                           data[4][i],
                                           data[5][i],
                                           data[6][i]))




    treeScroll = ttk.Scrollbar(frame_display, orient='vertical', command=tree.yview)
    tree.grid(row=0, column=0, sticky='nsew')
    treeScroll.grid(row=0, column=1, sticky='ns')
    tree.configure(yscroll=treeScroll.set) 

###################################################################################################################
#################################################################################################################
def window():
    root.destroy()


button1 = tk.Button(frame_alpr, text=" Load Data ", command=load_csv,width=13, height=1, font=('times', 13, ' bold '),bg="white",fg="red")
button1.place(x=12, y=50)


button2 = tk.Button(frame_alpr, text="PreProcess Data", command=process_data, width=13, height=1, font=('times', 13, ' bold '),bg="white",fg="red")
button2.place(x=12, y=120)

#direct_NB   cl_NB
button3 = tk.Button(frame_alpr, text="Naive Bayes", command=cl_NB, width=13, height=1, font=('times', 13, ' bold '),bg="white",fg="red")
button3.place(x=10, y=190)

#
button4 = tk.Button(frame_alpr, text="Prediction", command=load_model,width=13, height=1, font=('times', 13, ' bold '),bg="white",fg="red")
button4.place(x=10, y=260)

button5 = tk.Button(frame_alpr, text="Clear", command=Clear_entry,width=13, height=1, font=('times', 13, ' bold '),bg="white",fg="red")
button5.place(x=10, y=330)


exit = tk.Button(frame_alpr, text="Exit", command=window, width=13, height=1, font=('times', 13, ' bold '),bg="red",fg="white")
exit.place(x=10, y=480)



root.mainloop()
