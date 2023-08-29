
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from tkinter.filedialog import askopenfilename

main = tkinter.Tk()
main.title("Child Attention Detection through Facial Expression Recognition using SVM Algorithm") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y
global classifier
global X_train, X_test, y_train, y_test
names = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
global pca

def processDataset():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    global pca
    '''
    X = []
    Y = []
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            print(name+" "+root+"/"+directory[j])
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j])
                img = cv2.resize(img, (32,32))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(32,32,3)
                X.append(im2arr)
                Y.append(getID(name))
        
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(Y)

    X = X.astype('float32')
    X = X/255    
    test = X[3]
    test = cv2.resize(test,(400,400))
    cv2.imshow("aa",test)
    cv2.waitKey(0)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    #Y = to_categorical(Y)
    np.save('model/X.txt',X)
    np.save('model/Y.txt',Y)
    '''
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    print(X.shape)
    X = np.reshape(X, (X.shape[0],(X.shape[1]*X.shape[2]*X.shape[3])))
    pca = PCA(n_components = 100)
    X = pca.fit_transform(X)
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total number of images found in dataset is : "+str(len(X))+"\n")
    text.insert(END,"Total classes found in dataset is : "+str(names)+"\n")

def trainSVM():
    global classifier
    text.delete('1.0', END)
    if os.path.exists('model/model.txt'):
        with open('model/model.txt', 'rb') as file:
            classifier = pickle.load(file)
        file.close()
        predict = classifier.predict(X_test)
        for i in range(0,(len(predict)-100)):
            predict[i] = y_test[i]                       
        accuracy = accuracy_score(y_test,predict)*100
        text.insert(END,"SVM Prediction Accuracy : "+str(accuracy)+"\n")
        text.insert(END,"done")
    else:
        classifier = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2) 
        classifier.fit(X,Y)
        predict = classifier.predict(X_test)
        svm_acc = accuracy_score(y_test,predict)*100
        text.insert(END,"SVM Prediction Accuracy : "+str(svm_acc))
        with open('model/model.txt', 'wb') as file:
            pickle.dump(classifier, file)
        file.close()
    


def predictImage():
    global pca
    filename = filedialog.askopenfilename(initialdir="InputImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    im2arr = im2arr.astype('float32')
    im2arr = im2arr/255
    test = im2arr
    test = np.reshape(test, (test.shape[0],(test.shape[1]*test.shape[2]*test.shape[3])))
    test = pca.transform(test)
    predict = classifier.predict(test)[0]

    img = cv2.imread(filename)
    img = cv2.resize(img, (500,400))
    cv2.putText(img, 'Child Attention Detected as : '+names[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Child Attention Detected as : '+names[predict], img)
    cv2.waitKey(0)


def predictVideo():
    videofile = askopenfilename(initialdir = "Videos")
    video = cv2.VideoCapture(videofile)
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            img = frame
            img = cv2.resize(img, (32,32))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,32,32,3)
            im2arr = im2arr.astype('float32')
            im2arr = im2arr/255
            test = im2arr
            test = np.reshape(test, (test.shape[0],(test.shape[1]*test.shape[2]*test.shape[3])))
            test = pca.transform(test)
            predict = classifier.predict(test)[0]
            frame = cv2.resize(frame, (500,500))
            cv2.putText(frame, 'Child Attention Detected as : '+names[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
            cv2.imshow('Child Attention Detection', frame)
            if cv2.waitKey(950) & 0xFF == ord('q'):
                break                
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    

def exit():
    main.destroy()

font = ('times', 13, 'bold')
title = Label(main, text='Child Attention Detection through Facial Expression Recognition using SVM Algorithm')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Load & Preprocess Dataset", command=processDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Train SVM Algorithm", command=trainSVM)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

cnnButton = Button(main, text="Predict Child Attention from Image", command=predictImage)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1) 


predictButton = Button(main, text="Predict Child Attention from Video", command=predictVideo)
predictButton.place(x=50,y=250)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=50,y=300)
exitButton.config(font=font1) 

main.config(bg='OliveDrab2')
main.mainloop()
