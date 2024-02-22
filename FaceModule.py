import tkinter as tk
from tkinter import simpledialog, messagebox
import threading
import cv2
import numpy as np
import face_recognition
import sqlite3
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.setup_database()
        self.load_users()
        self.init_ui()
        self.start_camera_thread()

    def setup_database(self):
        self.conn = sqlite3.connect('users.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS users
                             (username TEXT PRIMARY KEY, encoding BLOB)''')
        self.conn.commit()

    def init_ui(self):
        self.video_label = tk.Label(self.root)
        self.video_label.pack()
        tk.Button(self.root, text="Register", command=self.register_user_gui).pack(fill=tk.X, padx=50, pady=5)

    def load_users(self):
        self.known_face_encodings = []
        self.known_face_usernames = []
        self.cursor.execute("SELECT username, encoding FROM users")
        for row in self.cursor.fetchall():
            username, encoding_blob = row
            encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            self.known_face_encodings.append(encoding)
            self.known_face_usernames.append(username)

    def start_camera_thread(self):
        threading.Thread(target=self.capture_faces, daemon=True).start()

    def capture_faces(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_usernames[first_match_index]

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                self.update_gui(frame)

    def update_gui(self, frame):
        """Update the GUI with a new frame."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def register_user_gui(self):
        username = simpledialog.askstring("Register", "Enter username to register:")
        if username:
            messagebox.showinfo("Action Needed", "Look at the camera for registration.")
            self.register_user(username)

    def register_user(self, username):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)
            if face_encodings:
                try:
                    encoding_blob = sqlite3.Binary(np.array(face_encodings[0]).tobytes())
                    self.cursor.execute("INSERT INTO users (username, encoding) VALUES (?, ?)", (username, encoding_blob))
                    self.conn.commit()
                    messagebox.showinfo("Success", f"User {username} registered successfully.")
                    self.load_users()
                except sqlite3.IntegrityError:
                    messagebox.showerror("Error", "Username already exists.")
            else:
                messagebox.showerror("Error", "No faces detected. Try again.")
        cap.release()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
