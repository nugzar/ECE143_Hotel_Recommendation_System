import random
import tkinter as tk
import pandas as pd

from data_cleaning import clean_data, get_user_booked
from weighted_prediction import FinalModel



##### DEFINING DATA DIRECTORIES
BOOKINGS_CSV = '../data/bookings.csv'
HOTELS_CSV = '../data/hotels.csv'
HOTEL_BOOKINGS_CSV = '../data/hotel_bookings.csv'


print("Importing data")
bookings = pd.read_csv(BOOKINGS_CSV)
hotels = pd.read_csv(HOTELS_CSV)
hotel_bookings = pd.read_csv(HOTEL_BOOKINGS_CSV)

print("Cleaning Data, and generating intermediate cleaned data")
df = clean_data(bookings,hotels , hotel_bookings)
user_watched = get_user_booked(df)

USER_IDS = user_watched["UserID"].unique().tolist()
DF_DATA = None


final = FinalModel(hotels, user_watched, df)
final.train()


def generate_dataframe(uid, weights):
    print(f"Getting prediction for user: {uid}, with weights: {weights} ")
    DF_DATA = final.predict(uid, weights).reset_index()
    return DF_DATA

class RecommenderUI:
    def __init__(self, master):
        self.master = master
        master.title("Hotel Recommender")

        # Initial Slider Values
        self.slider_1_value = 75
        self.slider_2_value = 15
        self.slider_3_value = 10

        # Weight Sliders
        self.slider_frame = tk.Frame(master)
        self.slider_frame.pack(fill=tk.X)

        self.slider_label_1 = tk.Label(self.slider_frame, text="Collaborative Filtering %")
        self.slider_label_1.pack(side=tk.LEFT)
        self.slider_1 = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                 variable=tk.DoubleVar(value=self.slider_1_value),
                                 command=self.update_sliders)
        self.slider_1.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.slider_label_2 = tk.Label(self.slider_frame, text="Monthly Trending %")
        self.slider_label_2.pack(side=tk.LEFT)
        self.slider_2 = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                 variable=tk.DoubleVar(value=self.slider_2_value),
                                 command=self.update_sliders)
        self.slider_2.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.slider_label_3 = tk.Label(self.slider_frame, text="Hotel Trending %")
        self.slider_label_3.pack(side=tk.LEFT)
        self.slider_3 = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                 variable=tk.DoubleVar(value=self.slider_3_value),
                                 command=self.update_sliders)
        self.slider_3.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # User ID Input
        self.input_frame = tk.Frame(master)
        self.input_frame.pack(fill=tk.X)

        self.input_label = tk.Label(self.input_frame, text="Enter a User ID:")
        self.input_label.pack(side=tk.LEFT)
        self.user_id_entry = tk.Entry(self.input_frame, validate="focusout", validatecommand=self.validate_user_id)
        self.user_id_entry.pack(side=tk.LEFT)
        self.user_id_entry.insert(0, str(USER_IDS[0]))

        # Random User Id Button
        self.random_user_button = tk.Button(self.input_frame, text="Random User ID", command=self.random_user_id)
        self.random_user_button.pack(side=tk.LEFT)

        # Button to predict
        self.button = tk.Button(master, text="Go", command=self.generate_dataframe)
        self.button.pack()

        # Displays Data
        self.output_frame = tk.Frame(master)
        self.output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_label = tk.Label(self.output_frame, text="Output")
        self.output_label.pack()

        self.output_table = pd.DataFrame(DF_DATA)
        self.output_table_widget = tk.Label(self.output_frame, text=self.output_table.to_string(index=False))
        self.output_table_widget.pack(fill=tk.BOTH, expand=True)

    def update_sliders(self, activeValue):
        total = 100
        active_value = int(activeValue)
        sliders = [self.slider_1, self.slider_2, self.slider_3]
        current_values = [slider.get() for slider in sliders]

        if sum(current_values) == total:
            return

        active_slider = -1
        try:
            active_slider = current_values.index(active_value)
        except ValueError:
            active_slider = 0

        inactive_slider_indices = {0, 1, 2} - {active_slider}
        remaining_total = total - active_value
        total_inactive = sum([current_values[i] for i in inactive_slider_indices])
        for slider_index in inactive_slider_indices:
            sliders[slider_index].set(round((remaining_total * current_values[slider_index]) / total_inactive))
        self.slider_1_value, self.slider_2_value, self.slider_3_value = (slider.get() for slider in sliders)

    def validate_user_id(self):
        user_id_str = self.user_id_entry.get()

        if user_id_str.isdigit() and int(user_id_str) in USER_IDS:
            self.user_id_entry.config(bg="white")
            return True
        else:
            self.user_id_entry.config(bg="red")
            return False

    def random_user_id(self):
        random_id = random.choice(USER_IDS)
        self.user_id_entry.delete(0, tk.END)
        self.user_id_entry.insert(0, str(random_id))
        self.user_id_entry.config(bg="white")

    def generate_dataframe(self):
        # Get the entered user_id or default to the first user_id if invalid
        try:
            user_id = int(self.user_id_entry.get())
            if user_id not in USER_IDS:
                raise ValueError()
        except ValueError:
            user_id = USER_IDS[0]
            self.user_id_entry.delete(0, tk.END)
            self.user_id_entry.insert(0, str(user_id))

        df = generate_dataframe(user_id, (self.slider_1_value, self.slider_2_value, self.slider_3_value))

        self.output_table = df
        self.output_table_widget.config(text=self.output_table.to_string(index=False))

# Start the application

root = tk.Tk()
app = RecommenderUI(root)
root.mainloop()
