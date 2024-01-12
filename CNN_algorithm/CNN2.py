#import libraries
import numpy as np
import pandas as pd
import tqdm as tqdm
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
import cv2
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Dense, Dropout, Conv2D
from keras.applications.vgg16 import VGG16  # Add this line

#r pour indiquer que la chaine est brute sinon on met \\
angry = r"C:\Users\zeine\OneDrive\Bureau\Data_Science_Project\CNN_algorithm\Dataset\train\angry"
disgust = r"C:\Users\zeine\OneDrive\Bureau\Data_Science_Project\CNN_algorithm\Dataset\train\disgust"
fear = r"C:\Users\zeine\OneDrive\Bureau\Data_Science_Project\CNN_algorithm\Dataset\train\fear"
happy = r"C:\Users\zeine\OneDrive\Bureau\Data_Science_Project\CNN_algorithm\Dataset\train\happy"
neutral = r"C:\Users\zeine\OneDrive\Bureau\Data_Science_Project\CNN_algorithm\Dataset\train\neutral"
sad = r"C:\Users\zeine\OneDrive\Bureau\Data_Science_Project\CNN_algorithm\Dataset\train\sad"
surprise = r"C:\Users\zeine\OneDrive\Bureau\Data_Science_Project\CNN_algorithm\Dataset\train\surprise"

#print number of each directory
print("Number of Images in Each Directory:")
print(f"Angry: {len(os.listdir(angry))}")
print(f"Disgust: {len(os.listdir(disgust))}")
print(f"fear: {len(os.listdir(fear))}")
print(f"happy: {len(os.listdir(happy))}")
print(f"neutral: {len(os.listdir(neutral))}")
print(f"sad: {len(os.listdir(sad))}")
print(f"surprise: {len(os.listdir(surprise))}")

#x and y are lists to save images
x = []
y = []
#dataset is list to save labels
dataset = []
#img_size is 256
img_size=48

#create lists to save the loss and the accuracy of each epoch
losses = []
accuracies = []
#list to save the 3 labels
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#Function to load images
#tqdm for visualize the progress of iterations
def get_data(directory, dir_name):
    for i in tqdm.tqdm(os.listdir(directory)):
        #combine directory path and name of image file
        full_path = os.path.join(directory, i)
        try:
            img = cv2.imread(full_path)
            img = cv2.resize(img, (48, 48))
        except:
            continue
        x.append(img)
        y.append(dir_name)
    return x, y

#function to pre_process data
def pre_process():
    x, y = get_data(angry, "angry")
    print(len(x), len(y))
    x, y = get_data(disgust, "disgust")
    print(len(x), len(y))
    x, y = get_data(fear, "fear")
    print(len(x), len(y))
    x, y = get_data(happy, "happy")
    print(len(x), len(y))
    x, y = get_data(neutral, "neutral")
    print(len(x), len(y))
    x, y = get_data(sad, "sad")
    print(len(x), len(y))
    x, y = get_data(surprise, "surprise")
    print(len(x), len(y))
    #convert x and y to an array
    x = np.array(x)  # array of images
    y = np.array(y)  # array of labels
    x.shape, y.shape
    ###############
    le = LabelEncoder()
    y = le.fit_transform(y)
    #random_state to fix random sequence to obtain reproductibles results
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #normalize pixels: convert image range 0..1
    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0

    x_train = x_train.reshape(-1, img_size, img_size, 3)
    y_train = np.array(y_train)

    x_test = x_test.reshape(-1, img_size, img_size, 3)
    y_test = np.array(y_test)
    ######################
    lb = LabelBinarizer()
    y_train_lb = lb.fit_transform(y_train)
    y_test_lb = lb.fit_transform(y_test)
    return x_test, y_test, y_test_lb, x_train, y_train, y_train_lb

#build model
# Build and train the VGG16-based model
def build_and_train_model(x_train, y_train_lb, x_test, y_test_lb, epochs=20, batch_size=32):
    model = VGG16(img_size, num_classes=7)

    checkpoint = ModelCheckpoint("vgg16_model.h5", monitor="val_accuracy", verbose=1, save_best_only=True,
                                 save_weights_only=False)

    earlystop = EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)

    for epoch in range(epochs):
        print("Epoch: ", epoch+1)
        history = model.fit(x_train, y_train_lb, epochs=1, validation_data=(x_test, y_test_lb),
                            batch_size=batch_size, verbose=1, callbacks=[checkpoint, earlystop])
        loss = history.history['loss'][0]
        accuracy = history.history['accuracy'][0]
        losses.append(loss)
        accuracies.append(accuracy)

    return model

# Evaluate the model
def performance(model, x_test, y_test_lb):
    loss, accuracy = model.evaluate(x_test, y_test_lb)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

# Make predictions
def predict(model, x_test, y_test):
    predict_x = model.predict(x_test)
    y_pred = np.argmax(predict_x, axis=1)

    accuracy = (y_pred == y_test).sum() / len(y_test)
    print("Model Accuracy:", accuracy)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    epoch_data = {'Epoch': range(1, len(losses) + 1), 'Loss': losses, 'Accuracy': accuracies}
    epoch_df = pd.DataFrame(epoch_data)

    with pd.ExcelWriter('vgg16_model_performance.xlsx') as writer:
        epoch_df.to_excel(writer, sheet_name='Epoch Data', index=False)

        accuracy_df = pd.DataFrame({'Accuracy': [accuracy]})
        accuracy_df.to_excel(writer, sheet_name='Accuracy', index=False)

        conf_matrix_df = pd.DataFrame(conf_matrix, columns=class_names, index=class_names)
        conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')

# Classify an image using the trained model
def classify_an_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48))
    img = np.reshape(img, [1, 48, 48, 3]) / 255.0
    class_probabilities = model.predict(img)
    predicted_class_id = np.argmax(class_probabilities)
    predicted_class = class_names[predicted_class_id]
    return predicted_class
