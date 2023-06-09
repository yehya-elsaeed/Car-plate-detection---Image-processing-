import cv2
import imutils
import pytesseract
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

class CarPlateDetectorGUI:



    def __init__(self):

        # Initialize the variables
        self.original_image = None
        self.plate_image = None
        self.characters_images = []
        self.characters_text = ""

        # Initialize tesseract
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        # setup
        self.window = tk.Tk()
        self.window.title("Car Plate Detector")
        self.window.geometry("1800x1000")
        self.window.minsize(800, 600)
        self.window.configure(bg='#000')

        # Create a label to show the original image
        self.original_label = tk.Label(self.window)
        self.original_label.pack()
        self.original_label.place(x=50, y=555)
        self.original_label.configure(bg='#000')

        # Create a label to show the original image
        self.original_label2 = tk.Label(self.window)
        self.original_label2.pack()
        self.original_label2.place(x=50, y=5)
        self.original_label2.configure(bg='#000')

        # Create a label to show the detected car plate image
        self.plate_label = tk.Label(self.window)
        self.plate_label.pack()
        self.plate_label.place(x=1000,y=100)
        self.plate_label.configure(bg='#000')

        # Create a label to show each character of the plate separated
        self.characters_label = tk.Label(self.window)
        self.characters_label.pack()
        self.characters_label.configure(bg='#000')
        self.characters_label.place(x=1000, y=300)

        # Create a text box to show the extracted characters as string
        self.text_box = tk.Text(self.window, height=3, width=60)
        self.text_box.pack()
        self.text_box.configure(bg='#fff')
        self.text_box.place(x=1000,y=15)

        # Create a button to open the car image file
        self.open_button = tk.Button(self.window,font="30",bg='#1f6aa5',fg='#FFF', text="Open Image", command=self.open_image)
        self.open_button.pack()
        self.open_button.place(x=1000,y=580)

        # Create a button to detect the car plate
        self.detect_button = tk.Button(self.window,font="30",bg='#1f6aa5',fg='#FFF', text="Detect Plate", command=self.detect_car_plate)
        self.detect_button.pack()
        self.detect_button.place(x=1000, y=620)

        # Create a button to show the characters of the plate
        self.show_characters_button = tk.Button(self.window,font="30",bg='#1f6aa5',fg='#FFF', text="Show Characters", command=self.show_characters)
        self.show_characters_button.pack()
        self.show_characters_button.place(x=1000,y=660)

        # Create a button to open a new car image file
        self.Show_Text_button = tk.Button(self.window,font="30",bg='#1f6aa5',fg='#FFF', text="Show Text", command=self.Show_Text)
        self.Show_Text_button.pack()
        self.Show_Text_button.place(x=1000,y=700)



    def open_image(self):

        # Show a file dialog to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        if file_path:
            # Load the selected image file and display it in the original label
            self.random_image = cv2.imread(file_path)
            self.original_image = imutils.resize(self.random_image,width=800,height=600 )

            # Clear the previous outputs
            self.clear_output()
            self.display_image(self.original_label2, self.original_image)


    def detect_car_plate(self):

        if self.original_image is not None:
            # Resize the image
            image = imutils.resize(self.original_image, width=min(500, len(self.original_image[0])))

            # RGB to Gray conversion
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Noise removal
            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            # Edge Detection in image
            edged = cv2.Canny(gray, 170, 200)

            # Find contours
            cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Create a copy of the RGB image to draw all contours
            img1 = image.copy()
            cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)

            # Sort contours based on area keeping minimum required area as '30'
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
            NumberPlateCnt = None  # We do not have no number plate contour

            # Top 30 contours are extracted
            img2 = image.copy()
            cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)

            # Finding the best possible approx contour of number plate
            count = 0
            idx = 7
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # Output number of edge in contour
                if len(approx) == 4:  # Contour with 4 corners
                    NumberPlateCnt = approx  # Our approx number plate contour

                    # Crop contours and store it into cropped images folder
                    x, y, w, h = cv2.boundingRect(c)  # Find co-ord for plate
                    plate_image = image[y:y + h, x:x + w]  # Create new image
                    self.plate_image = imutils.resize(plate_image,width=400,height=300)
                    self.display_image(self.plate_label, self.plate_image)
                    idx += 1
                    break

            # Drawing selected contour in org image
            cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
            self.display_image(self.original_label, image)
            self.display_image(self.original_label2, self.original_image)

    def show_characters(self):
        if self.plate_image is not None:
            # Use tesseract to convert each character image into string
            self.characters_images = []
            self.characters_text = ""
            gray = cv2.cvtColor(self.plate_image, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                character_image = self.plate_image[y:y + h, x:x + w]
                # Extract character using Tesseract OCR
                character_text = pytesseract.image_to_string(self.plate_image, lang='eng', config='--psm 10')
                self.characters_text = character_text
                self.characters_images.append(character_image)


            # Display the extracted characters as a string in the text box
            self.text_box.delete('1.0', tk.END)
            self.text_box.insert(tk.END, self.characters_text)

            # Display each character image in the characters label
            self.display_characters()

    def Show_Text(self):
        character_image = self.plate_image
        # cv2.imshow( "okay", character_image)
        character_text = pytesseract.image_to_string(character_image, lang='eng')
        self.text_box.insert(tk.END, character_text)

    def display_image(self, label, image):
        # Convert the image to a tkinter PhotoImage and display it in the given label
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        label.configure(image=image)
        label.image = image

    def display_characters(self):

        # Display each character image in the characters label
        images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in self.characters_images]
        widths, heights = zip(*(i.size for i in images))
        max_width_index = widths.index(max(widths))
        widths = list(widths)
        images = list(images)
        total_width = sum(widths)
        max_height = max(heights)
        del widths[max_width_index]
        del images[max_width_index]
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for image in images:
            new_im.paste(image, (x_offset, 0))
            x_offset += image.size[1]
        new_im = ImageTk.PhotoImage(new_im)
        self.characters_label.configure(image=new_im)
        self.characters_label.image = new_im

    def clear_output(self):
        # Create a black image
        black_image = Image.new("RGB", (100, 100), color="black")
        black_photo = ImageTk.PhotoImage(black_image)

        # Set the labels to display the black image
        self.original_label.configure(image=black_photo, text="")
        self.plate_label.configure(image=black_photo, text="")
        self.characters_label.configure(image=black_photo, text="")

        # Clear the text box
        self.text_box.delete("1.0", tk.END)
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    gui = CarPlateDetectorGUI()
    gui.run()