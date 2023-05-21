import tkinter as tk
from tkinter import filedialog
from tkinter import *

from PIL import ImageTk, Image
import cv2

import numpy as np

from keras.models import load_model

# Load mô hình đã được huấn luyện
model = load_model('vehicle_classification_model.h5')

# Danh sách các lớp phương tiện giao thông
classes = { 
    0:'AMBULANCE',
    1:'BIKE',
    2:'BUS',
    3:'CAR',
    4:'FIRE TRUCK',
    5:'MOTOBIKE',
    6:'AIRPLANE',
    7:'SHIP',
    8:'TRAIN',
    9:'TRUCK'
}

# Hàm sử dụng cho chế độ Camera
def classify_CAM(image):
    # Tiền xử lý ảnh
    image = cv2.resize(image, (100, 100))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image.astype('float32') / 255
    # Dự đoán
    pred_np = model.predict(image)
    pred = np.argmax(pred_np)   
    accuracy = pred_np[0][pred]
    # Chuyển sang dạng string
    sign = "Classify: " + classes[pred]
    accuracy = "Accuracy: " + str(round(accuracy*100)) +"%"
    return sign, accuracy

# Hàm đặt chữ lên ảnh
def put_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 255), thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness)

# Hàm chạy chế độ Camera
def RunCam():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)

    while True:
        _, img = cap.read()  

        # Xử lý ảnh
        label, acc = classify_CAM(img)
        # Hiển thị kết quả
        put_text(img, label, (10, 50))
        put_text(img, acc, (10, 80))

        # Xuất ảnh
        cv2.imshow('Original', img)
        flag = cv2.waitKey(1)

        if flag == ord('q') or flag == ord('Q'): 
            break

    cv2.destroyAllWindows()

# Hàm sử dụng cho giao diện GUI
def classify_GUI(file_path):
    try:    
        # Tiền xử lý ảnh
        image = cv2.imread(file_path)
        image = cv2.resize(image, (100,100))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        image = image/255
        # Dự đoán
        pred_np = model.predict(image)
        pred = np.argmax(pred_np)
        #Chuyển sang dạng string
        sign = classes[pred]

        label.configure(foreground='#011638', text=sign)
    except: 
        label.configure(text='Phải chọn ảnh có đủ 3 kênh màu R G B', foreground='#FF0000')

# Hiển thị nút "Classify Image" sau khi đã chọn ảnh
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify_GUI(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.5, rely=0.8, anchor=CENTER)

# Tải ảnh lên
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        label.configure(text='Lỗi khi mở tệp vừa chọn, thử lại', foreground='#FF0000')

# Đóng cửa sổ khi nhấn phím "Q" hoặc "q"
def close_window(event):
    if event.keysym.lower() == 'q' or event.keysym.lower() == 'Q':
        top.destroy()

# Tạo cửa sổ giao diện GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Vehicle Classification')


# Tạo ảnh nền
image = Image.open('BACKGROUND.jpg')
resized_image = image.resize((800, 600), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(resized_image)
background_label = tk.Label(top, image=background_image)
background_label.place(relx=0, rely=0)

label=Label(top, background='#FFFFFF', font=('arial',15,'bold'))
label.place(relx=0.5, rely=0.9, anchor=CENTER)

sign_image = Label(top)
sign_image.place(relx=0.5, rely=0.5, anchor=CENTER)

use_image=Button(top,text="Use Image",command=upload_image,padx=10,pady=5)
use_image.configure(background='#364156', foreground='#FFFFFF',font=('arial',10,'bold'))
use_image.place(relx=0.8, rely=0.5, anchor=W)

use_camera=Button(top,text="Use Camera",command=RunCam,padx=10,pady=5)
use_camera.configure(background='#364156', foreground='#FFFFFF',font=('arial',10,'bold'))
use_camera.place(relx=0.2, rely=0.5, anchor=E)

heading = Label(top, text="Vehicle Classification", pady=20, font=('arial',20,'bold'))
heading.configure(background='#FFFFFF',foreground='#364156')
heading.place(relx=0.5, rely=0.2, anchor=CENTER)

# Gắn sự kiện đóng cửa sổ
top.bind('<Key>', close_window)
top.mainloop()