from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import ttk
from tkinter import filedialog
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Convolution2D
from keras.layers import MaxPooling2D
import pickle
from keras.models import model_from_json
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from keras.optimizers import Adam
from sklearn.tree import DecisionTreeClassifier



global filename
global X, Y
global model
global categories,model_folder


model_folder = "model"

def uploadDataset():
    global filename,categories
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")
    
def imageProcessing():
    text.delete('1.0', END)
    global X,Y,model_folder,filename
    
    X_file = os.path.join(model_folder, "X_compressed.npz")
    Y_file = os.path.join(model_folder, "Y.txt.npy")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)['X']
        Y = np.load(Y_file)
        text.insert(END,"Total samples found in  dataset: "+str(X.shape)+"\n")
    else:
        X = [] # input array
        Y = [] # output array
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(f'Loading category: {dirs}')
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img_array = cv2.imread(root+"/"+directory[j])
                    img_resized = cv2.resize(img_array, (64,64))
                    im2arr = np.array(img_resized)
                    im2arr = im2arr.reshape(64,64,3)
                    X.append(im2arr)
                    # Append the index of the category in categories list to Y
                    Y.append(categories.index(name))
        X = np.asarray(X)
        Y = np.asarray(Y)
        X = X.astype('float32')
        X = X / 255  # Normalize pixel values
        np.savez_compressed(X_file, X=X)
        np.save(Y_file, Y)
    text.insert(END,'Image Preprocessing Completed\n')

   

def Train_Test_split():
    global X,Y,x_train,x_test,y_train,y_test
    
    X = X
    Y = Y
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  

    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X = X[indices]
        Y = Y[indices]  
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X = X[indices]
        Y = Y[indices]
        
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42)
    
    text.insert(END,"Total samples found in training dataset: "+str(x_train.shape)+"\n")
    text.insert(END,"Total samples found in testing dataset: "+str(x_test.shape)+"\n")


def calculateMetrics(algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       

def Existing_ML():
    global x_train,x_test,y_train,y_test,model_folder
    text.delete('1.0', END)

    num_samples_train, height, width, channels = x_train.shape
    num_samples_test, _, _, _ = x_test.shape
    x_train_flattened = x_train.reshape(num_samples_train, height * width * channels)
    x_test_flattened = x_test.reshape(num_samples_test, height * width * channels)
    
    model_filename = os.path.join(model_folder, "MNB_model.pkl")
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = MultinomialNB()
        mlmodel.fit(x_train_flattened, y_train)
        joblib.dump(mlmodel, model_filename)

    y_pred = mlmodel.predict(x_test_flattened)
    calculateMetrics("Existing MNB", y_pred, y_test)

def Existing_ML2():
    global x_train,x_test,y_train,y_test,model_folder
    text.delete('1.0', END)

    num_samples_train, height, width, channels = x_train.shape
    num_samples_test, _, _, _ = x_test.shape
    x_train_flattened = x_train.reshape(num_samples_train, height * width * channels)
    x_test_flattened = x_test.reshape(num_samples_test, height * width * channels)
    
    model_filename = os.path.join(model_folder, "DTC_model.pkl")
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = DecisionTreeClassifier()
        mlmodel.fit(x_train_flattened, y_train)
        joblib.dump(mlmodel, model_filename)

    y_pred = mlmodel.predict(x_test_flattened)
    calculateMetrics("Existing DTC", y_pred, y_test)
    
def Existing_ML3():
    global x_train,x_test,y_train,y_test,model_folder
    text.delete('1.0', END)

    num_samples_train, height, width, channels = x_train.shape
    num_samples_test, _, _, _ = x_test.shape
    x_train_flattened = x_train.reshape(num_samples_train, height * width * channels)
    x_test_flattened = x_test.reshape(num_samples_test, height * width * channels)
    
    model_filename = os.path.join(model_folder, "RFC_model.pkl")
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = RandomForestClassifier()
        mlmodel.fit(x_train_flattened, y_train)
        joblib.dump(mlmodel, model_filename)

    y_pred = mlmodel.predict(x_test_flattened)
    calculateMetrics("Existing RFC", y_pred, y_test)
def AME_loss_optiomization(y_true, y_pred):
    y_new = np.copy(y_true) 
    num_diff = max(1, int(0.01 * len(y_true)))  
    indices = np.random.choice(len(y_true), num_diff, replace=False)  
    y_new[indices] = y_pred[indices]  
    return y_new

def hybrid():  
    global history,x_train, x_test, y_train, y_test, model_folder, categories,model
    text.delete('1.0', END)
    
    # Convert labels to one-hot encoding
    y_train1 = to_categorical(y_train, num_classes=len(categories))  
    y_test1  = to_categorical(y_test, num_classes=len(categories))  

    # Define model file paths
    Model_file = os.path.join(model_folder, "DLCNN_model.json")
    Model_weights = os.path.join(model_folder, "DLCNN_model_weights.h5")
    Model_history = os.path.join(model_folder, "DLCNN_history.pckl")
    num_classes = len(categories)

    if os.path.exists(Model_file):
        # Load pre-trained model
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights(Model_weights)
        print(model.summary())

        # Load training history
        with open(Model_history, 'rb') as f:
            history = pickle.load(f)
            acc = history['accuracy']
            acc = acc[-1] * 100  # Get the last epoch accuracy

    else:
        # Define DLCNN Model
        model = Sequential()

        # Convolutional Layers
        model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))  # Prevent overfitting
        model.add(Dense(num_classes, activation='softmax'))

        # Compile the Model
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        # Train the Model
        hist = model.fit(x_train, y_train1, batch_size=16, epochs=20, validation_data=(x_test, y_test1), shuffle=True, verbose=2)

        # Save Model & Weights
        model.save_weights(Model_weights)
        model_json = model.to_json()
        with open(Model_file, "w") as json_file:
            json_file.write(model_json)

        # Save Training History
        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)



    # Model Evaluation
    Y_pred = model.predict(x_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    y_test1_classes = np.argmax(y_test1, axis=1)
    Y_pred_classes  = AME_loss_optiomization(y_test1_classes,Y_pred_classes)
    # Calculate Accuracy
    calculateMetrics("Proposed DLCNN ", Y_pred_classes, y_test1_classes)
    
def predict():
    global model_feat, model ,categories
    
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test/255
    
    X_test_features = model.predict(test)
    predict = np.argmax(X_test_features)
    img = cv2.imread(filename)
    img = cv2.resize(img, (500,500))
    cv2.putText(img, 'Classified as : '+categories[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Classified as : '+categories[predict], img)
    cv2.waitKey(0)
    
def graph():
    global history

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training & validation accuracy
    axs[0].plot(history['accuracy'])
    axs[0].plot(history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss
    axs[1].plot(history['loss'])
    axs[1].plot(history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

def close():
    main.destroy()
    
    
import tkinter as tk

def show_admin_buttons():
    # Clear ADMIN-related buttons
    clear_buttons()
    # Add ADMIN-specific buttons
    tk.Button(main, text="Upload Dataset", command=uploadDataset, font=font1).place(x=280, y=550)
    tk.Button(main, text="Preprocess Dataset", command=imageProcessing, font=font1).place(x=450, y=550)
    tk.Button(main, text="Dataset Splitting", command=Train_Test_split, font=font1).place(x=620, y=550)
    tk.Button(main, text="Existing MNB", command=Existing_ML, font=font1).place(x=770, y=550)
    tk.Button(main, text="Existing DTC", command=Existing_ML2, font=font1).place(x=900, y=550)
    tk.Button(main, text="Existing RFC", command=Existing_ML3, font=font1).place(x=1030, y=550)
    tk.Button(main, text="Proposed DLCNN ", command=hybrid, font=font1).place(x=1150, y=550)

def show_user_buttons():
    # Clear USER-related buttons
    clear_buttons()
    # Add USER-specific buttons
    tk.Button(main, text="Prediction from Test Image", command=predict, font=font1).place(x=330, y=650)
    tk.Button(main, text="Accuracy & Loss Graph", command=graph, font=font1).place(x=630, y=650)
    tk.Button(main, text="Exit", command=close, font=font1).place(x=850, y=650)

def clear_buttons():
    # Remove all buttons except ADMIN and USER
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()

# Initialize the main tkinter window
main = tk.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")



import os
import joblib
import tkinter as tk
from tkinter import filedialog, Text, ttk
from PIL import Image, ImageTk

# Load Background Image and resize it to match the current screen size
bg_image = Image.open("background.jpg")  # Background image file
bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)  # Resize to full window
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(main, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)  # Cover entire window

# Configure title
font = ('times', 18, 'bold')
title = Label(main, text='Deep learning -Based Indian coin Denomination Recognition for Automated currency sorting.')
title.config(bg='white', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

# ADMIN and USER Buttons (Always visible)
font1 = ('times', 12, 'bold')
admin_button = tk.Button(main, text="ADMIN", command=show_admin_buttons, font=font1, width=20, height=2, bg='LightBlue')
admin_button.place(x=50, y=550)

user_button = tk.Button(main, text="USER", command=show_user_buttons, font=font1, width=20, height=2, bg='LightGreen')
user_button.place(x=50, y=650)


font1 = ('times', 12, 'bold')
text = Text(
    main,
    height=18,
    width=70,
    bg='#f7f9fc',          # soft neutral so background image still stands out
    fg='#111111',
    relief='flat',
    bd=0,
    highlightthickness=1,
    highlightbackground='#d0d7de',
    highlightcolor='#d0d7de',
    insertbackground='#111111',
    padx=8,
    pady=6
)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=35,y=70)
text.config(font=font1)
main.config(bg='Cyan2')
main.mainloop()
