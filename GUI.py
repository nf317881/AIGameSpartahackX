from copy import copy
import math
import tkinter as tk
from tkinter import font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter.simpledialog as simpledialog  # Import simpledialog for user input
import funcLibrary

model_list = []

class Level:
    def __init__(self, master, back_callback, score, classification, next_frame, func = None, MNIST_images = None):
        self.master = master
        self.back_callback = back_callback
        self.next_frame = next_frame
        self.dragging = None

        self.classification = classification
        self.func = func
        self.MNIST_images = MNIST_images
        self.score = score
        
        # Configure main frame
        self.frame = tk.Frame(master, bg="#1a1a1a")
        self.frame.pack(fill="both", expand=True)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.frame.columnconfigure(2, weight=1)
        self.frame.rowconfigure(0, weight=1)
        
        # Create layout sections
        self.create_graph_section()
        self.create_toolbox()
        self.create_palette()
        self.create_control_buttons()

    def display_digits(self, digits_to_display):
                """
                Clears the current graph canvas and displays the specified digits in a grid.
                :param digits_to_display: List of digits (e.g. [0, 1, 2, 3]) to display. 
                                        If empty, the canvas will be cleared.
                """
                # Get the parent widget of the current graph canvas
                graph_parent = self.canvas.get_tk_widget().master
                # Destroy the existing canvas widget to clear the old graph
                self.canvas.get_tk_widget().destroy()

                count = len(digits_to_display)
                if count == 0:
                    # If no digits are provided, simply create an empty figure with the desired background.
                    fig = plt.figure(facecolor="#2c3e50")
                else:
                    # Determine grid dimensions (fixed columns, rows computed from count)
                    columns = 5
                    rows = math.ceil(count / columns)
                    fig, axes = plt.subplots(int(rows), int(columns), figsize=(4, 4), facecolor="#2c3e50")
                    # Ensure axes is a flat array even if there's only one row
                    axes = np.array(axes).flatten()
                    for i, ax in enumerate(axes):
                        if i < count:
                            # Display the digit in the center of the subplot
                            ax.text(0.5, 0.5, str(digits_to_display[i]), fontsize=32, ha="center", va="center", color="white")
                        # Turn off the axis for a cleaner look
                        ax.axis("off")
                        ax.set_facecolor("#2c3e50")
                # Create a new canvas for the updated figure and attach it to the same parent widget
                self.canvas = FigureCanvasTkAgg(fig, master=graph_parent)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_graph_section(self):
        graph_frame = tk.Frame(self.frame, bg="#2c3e50", width=400, height=400)
        graph_frame.grid(row=0, column=0, rowspan=1, padx=10, pady=10, sticky="nsew")
        fig, ax = plt.subplots(figsize=(4, 4), facecolor="#2c3e50")
        x = np.linspace(-5, 5, 100)
        ax.plot(x, self.func(x), color="#3498db")
        ax.set_facecolor("#2c3e50")
        ax.tick_params(colors="white")
        self.canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        if self.classification:
            digits_to_display = []
            for x in range(len(self.MNIST_images)):
                if self.MNIST_images[x]:
                    digits_to_display.append(x)

            self.display_digits(digits_to_display)

    def create_toolbox(self):
        # Toolbox with the requested components
        self.toolbox_frame = tk.Frame(self.frame, bg="#2c3e50", width=150, height=1000)
        self.toolbox_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        components = [
            "Linear", "Dropout", "ReLU", "Leaky ReLU", "ELU", 
            "Sigmoid", "Batch Normalization", "Convolutional", 
            "Pooling", "Flatten"
        ]
        for i, comp in enumerate(components):
            btn = tk.Label(self.toolbox_frame, text=comp, bg="#3498db", fg="white",
                           font=("Arial", 10), padx=10, pady=5)
            btn.grid(row=i, column=0, pady=5, padx=5, sticky="ew")
            btn.bind("<Button-1>", lambda e, c=comp: self.start_drag(e, c))
            btn.bind("<B1-Motion>", self.on_drag)
            btn.bind("<ButtonRelease-1>", self.stop_drag)

    def create_palette(self):
        # Palette takes up a fixed area (adjust as desired)
        self.palette_frame = tk.Frame(self.frame, bg="#34495e", width=400, height=600)
        self.palette_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.palette_frame.grid_propagate(False)  # Fix size
        self.palette_frame.columnconfigure(0, weight=1)
        tk.Label(self.palette_frame, text="Palette", bg="#34495e", fg="white",
                 font=("Arial", 12)).grid(row=0, column=0, pady=5)


    def create_control_buttons(self):
        btn_frame = tk.Frame(self.frame, bg="#1a1a1a")
        btn_frame.grid(row=1, column=0, columnspan=3, pady=10)
        tk.Button(btn_frame, text="Train", bg="#27ae60", fg="white",
                  command=self.train_model).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Reset", bg="#e74c3c", fg="white",
                  command=self.reset_network).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Back", bg="#7f8c8d", fg="white",
                  command=self.back_callback).pack(side="left", padx=5)

    def start_drag(self, event, component):
        x = event.x_root - self.master.winfo_rootx()
        y = event.y_root - self.master.winfo_rooty()
        drag_label = tk.Label(self.master, text=component, bg="#2980b9", fg="white",
                              font=("Arial", 10), padx=10, pady=5)
        drag_label.place(x=x, y=y)
        self.dragging = {"component": component, "x": event.x_root, "y": event.y_root, "widget": drag_label}

    def on_drag(self, event):
        if self.dragging:
            dx = event.x_root - self.dragging["x"]
            dy = event.y_root - self.dragging["y"]
            widget = self.dragging["widget"]
            new_x = widget.winfo_x() + dx
            new_y = widget.winfo_y() + dy
            widget.place(x=new_x, y=new_y)
            self.dragging["x"] = event.x_root
            self.dragging["y"] = event.y_root

    def stop_drag(self, event):
        global model_list

        if self.dragging:
            palette_x = self.palette_frame.winfo_rootx()
            palette_y = self.palette_frame.winfo_rooty()
            palette_w = self.palette_frame.winfo_width()
            palette_h = self.palette_frame.winfo_height()
            if (palette_x <= event.x_root <= palette_x + palette_w and
                palette_y <= event.y_root <= palette_y + palette_h):
                comp = self.dragging["component"]
                # Optionally, determine the next row (skip header row 0)
                current_items = self.palette_frame.grid_slaves()
                row = len(current_items)
                
                # If the component requires parameters, prompt the user.
                if comp in ["Linear", "Pooling", "Convolutional"]:
                    params = self.prompt_for_params(comp)
                    if params is None:
                        # If user cancels input, do not add the component.
                        self.dragging["widget"].destroy()
                        self.dragging = None
                        return
                    if comp == "Linear":
                        display_text = f"{comp}\n(neurons: {params})"
                    elif comp == "Pooling":
                        display_text = f"{comp}\n(kernel: {params})"
                    elif comp == "Convolutional":
                        display_text = f"{comp}\n(channels: {params[0]}, kernel: {params[1]})"
                else:
                    display_text = comp
                new_label = tk.Label(self.palette_frame, text=display_text, bg="#3498db", fg="white",
                                     font=("Arial", 10), padx=10, pady=5, anchor="center")
                new_label.grid(row=row, column=0, pady=5, padx=5, sticky="ew")

                layer_list = [comp]
                try: 
                    if type(params) != int:
                        for x in params:
                            layer_list.append(x)
                    else:
                        layer_list.append(params)
                except:
                    pass
                model_list.append(layer_list)

            self.dragging["widget"].destroy()
            self.dragging = None

    def prompt_for_params(self, comp):
        """Prompt the user for parameters depending on the component type."""
        if comp == "Linear":
            neurons = simpledialog.askinteger("Linear Parameter",
                                               "Enter neuron amount (1-1000):",
                                               parent=self.master,
                                               minvalue=1, maxvalue=1000)
            return neurons
        elif comp == "Pooling":
            kernel = simpledialog.askinteger("Pooling Parameter",
                                             "Enter kernel size (2-10):",
                                             parent=self.master,
                                             minvalue=2, maxvalue=10)
            return kernel
        elif comp == "Convolutional":
            channels = simpledialog.askinteger("Convolutional Parameter",
                                               "Enter output channel amount (1-1000):",
                                               parent=self.master,
                                               minvalue=1, maxvalue=1000)
            kernel = simpledialog.askinteger("Convolutional Parameter",
                                             "Enter kernel size (2-10):",
                                             parent=self.master,
                                             minvalue=2, maxvalue=10)
            return (channels, kernel)

    def train_model(self):
        # Build a list of layers from the palette.
        global model_list

        # Create the neural network model using your makemodel() function.
        # (Make sure to import makemodel from its module at the top of your file.)
        model = funcLibrary.makemodel(model_list)
        print("Neural network created:")
        print(model)

        train_time = simpledialog.askinteger("Train Model",
                                         "Enter training time (in seconds):",
                                         parent=self.master,
                                         minvalue=1)
        
        self.train_time = train_time
        if train_time is None:
            # User cancelled, so return without training.
            self.inaccuracy = 3E8
            self.total_error = 3E8
            self.std_error = 3E8
            return 
        # For demonstration, we simply print the time.
        # Replace the following line with your actual training logic.
        print(f"Training model for {train_time} seconds...")

        MNIST_copy = copy(self.MNIST_images)

        if not self.classification:
            x_train, x_test, y_train, y_test = funcLibrary.generate_data(self.func)
        else:
            for x in self.MNIST_images:
                if x == 10:
                    MNIST_copy[MNIST_copy.index(x)] = (8, 2) #apple
                else:
                    MNIST_copy[MNIST_copy.index(x)] = (0, 0)

            x_train, y_train, x_test, y_test = funcLibrary.get_custom_mnist(MNIST_copy[0], MNIST_copy[1], MNIST_copy[2],
                    MNIST_copy[3], MNIST_copy[4], MNIST_copy[5], MNIST_copy[6], MNIST_copy[7], MNIST_copy[8], MNIST_copy[9])


        if self.classification:
            self.inaccuracy = funcLibrary.train_classification_model_time_based(model, x_train, y_train, x_test, y_test, time_limit=train_time)
        else:
            self.total_error, self.std_error = funcLibrary.train_and_test_model_time_based(model, x_train, y_train, x_test, y_test, time_limit=train_time)
        self.calculate_score()

    def calculate_score(self):
        # Create score popup window
        self.score_window = tk.Toplevel(self.master)
        self.score_window.title("Level Results")
        self.score_window.geometry("400x300")
        self.score_window.configure(bg="#1a1a1a")
        self.score_window.grab_set()  # Make it modal

        # Create rounded rectangle background
        canvas = tk.Canvas(self.score_window, bg="#1a1a1a", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Determine star ratings based on performance
        if self.classification:
            score = self.inaccuracy*(self.train_time+1)
            stars = 3 if score < self.score else self.score*1.25 if score < 50 else self.score*1.5
        else:
            score = self.total_error*(self.train_time+1)
            stars = 3 if score < self.score else self.score*1.25 if score < 50 else self.score*1.5

        # Create stars
        star_frame = tk.Frame(canvas, bg="#2c3e50")
        star_frame.place(relx=0.5, rely=0.3, anchor="center")
        
        for i in range(3):
            color = "#f1c40f" if i < stars else "#7f8c8d"
            tk.Label(star_frame, text="★", font=("Arial", 32), 
                    fg=color, bg="#2c3e50").pack(side="left", padx=5)

        # Score display
        score_text = f"Error: {score:.2f}" if not self.classification else f"Inaccuracy: {score:.1f}%"
        tk.Label(canvas, text=score_text, font=("Arial", 16), 
                fg="white", bg="#2c3e50").place(relx=0.5, rely=0.5, anchor="center")

        # Buttons
        btn_frame = tk.Frame(canvas, bg="#2c3e50")
        btn_frame.place(relx=0.5, rely=0.8, anchor="center")
        
        tk.Button(btn_frame, text="Retry", bg="#e74c3c", fg="white",
                command=lambda: [self.reset_network(), self.score_window.destroy()]).pack(side="left", padx=10)
        
        # Only show next level button if stars > 1
        if stars > 1:
            tk.Button(btn_frame, text="Next Level", bg="#27ae60", fg="white",
                    command=self.next_level).pack(side="left", padx=10)

    def next_level(self):
        self.score_window.destroy()
        self.next_frame()
        print("Proceeding to next level...")


    def reset_network(self):
        global model_list
        # Reset the palette: destroy all children except the header (assumed at row 0)
        for widget in self.palette_frame.grid_slaves():
            if int(widget.grid_info()["row"]) != 0:
                widget.destroy()
        model_list = []

current_level = 0

class GameLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Main Menu")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")

        self.colors = {
            "bg": "#1a1a1a",
            "button_bg": "#4d4d4d",
            "button_active": "#666666",
            "text": "#ffffff",
            "hover_bg": "#666666"
        }
        
        self.main_frame = tk.Frame(root, bg="#1a1a1a")
        self.credits_frame = tk.Frame(root, bg="#1a1a1a")
        self.levels_frame = tk.Frame(root, bg="#1a1a1a")
        self.tutorial_frame = tk.Frame(root, bg="#1a1a1a")
        self.resources_frame = tk.Frame(root, bg="#1a1a1a")

        self.level_frames = []
        for x in range(10):
            self.level_frames.append(tk.Frame(root, bg="#1a1a1a"))
        
        self.create_main_menu()
        self.create_credits_screen()
        self.create_levels_screen()
        self.create_tutorial_screen()
        self.create_each_level()
        self.create_resources_screen()

        self.hide_all_frames()
        self.show_main_menu()
        

        
    def show_tutorial(self):
        self.hide_all_frames()
        self.tutorial_frame.place(relx=0.5, rely=0.5, anchor="center")

    def start_level(self, num):
        global current_level

        current_level = num
        self.hide_all_frames()
        self.level_frames[num-1].place(relx=0.5, rely=0.5, anchor="center")

    def next_level(self):
        global current_level

        if current_level < 10:
            self.start_level(current_level+1)
            current_level+=1
        else:
            self.show_main_menu()

    def create_each_level(self):
        self.levels = []
        self.levels.append(Level(self.level_frames[9], self.show_levels, 3E8, True, self.next_level, func = lambda x: abs(x), MNIST_images=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10])) #10
        self.levels.append(Level(self.level_frames[8], self.show_levels, 3E8, True, self.next_level, func = lambda x: abs(x), MNIST_images=[10, 0, 0, 10, 0, 10, 0, 0, 10 ,0])) #9
        self.levels.append(Level(self.level_frames[7], self.show_levels, 3E8, True, self.next_level, func = lambda x: abs(x), MNIST_images=[0, 0, 0, 10, 0, 0, 0, 0, 10 ,0])) #8
        self.levels.append(Level(self.level_frames[6], self.show_levels, 3E8, True, self.next_level, func = lambda x: abs(x), MNIST_images=[10, 10, 10, 0, 0, 0, 0, 0, 0 ,0])) #7
        self.levels.append(Level(self.level_frames[5], self.show_levels, 3E8, True, self.next_level, func = lambda x: abs(x), MNIST_images=[10, 10, 0, 0, 0, 0, 0, 0, 0 ,0])) #6
        self.levels.append(Level(self.level_frames[4], self.show_levels, 3E8, False, self.next_level, func = lambda x: np.sin(x))) #5
        self.levels.append(Level(self.level_frames[3], self.show_levels, 3E8, False, self.next_level, func = lambda x: x**2)) #4
        self.levels.append(Level(self.level_frames[2], self.show_levels, 3E8, False, self.next_level, func = lambda x: np.log(x+6))) #3
        self.levels.append(Level(self.level_frames[1], self.show_levels, 3E8, False, self.next_level, func = lambda x: np.sqrt(x+5.1))) #2  
        self.levels.append(Level(self.level_frames[0], self.show_levels, 3E8, False, self.next_level, func = lambda x: abs(x))) #1
        self.levels.reverse()
        

    def create_tutorial_screen(self):
        self.tutorial = Level(self.tutorial_frame, self.show_main_menu, 3E8, False, self.show_main_menu, func = lambda x: x)

    def hide_all_frames(self):
        self.main_frame.place_forget()
        self.credits_frame.place_forget()
        self.levels_frame.place_forget()
        self.tutorial_frame.place_forget()
        self.resources_frame.place_forget()
        for x in range(10):
            self.level_frames[x].place_forget()

    def create_main_menu(self):
        self.button_font = font.Font(family="Arial", size=18, weight="bold")
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")
        for idx, (text, command) in enumerate([
                ("Levels", self.show_levels),
                ("Tutorial", self.show_tutorial),
                ("Resources", self.show_resources),
                ("Credits", self.show_credits)]):
            btn = tk.Button(self.main_frame, text=text, font=self.button_font, 
                            bg="#4d4d4d", fg="#ffffff", activebackground="#666666", 
                            activeforeground="#ffffff", relief="flat", borderwidth=0, 
                            width=15, command=command)
            btn.grid(row=idx, column=0, pady=10, ipadx=10, ipady=8)

    def create_levels_screen(self):
        global current_level

        # Level buttons grid
        for level in range(1, 11):
            row = (level-1) // 5
            col = (level-1) % 5
            
            btn = tk.Button(
                self.levels_frame,
                text=f"Level {level}",
                font=("Arial", 18, "bold"),
                bg=self.colors["button_bg"],
                fg=self.colors["text"],
                activebackground=self.colors["button_active"],
                activeforeground=self.colors["text"],
                relief="flat",
                width=10,
                command=lambda l=level: self.start_level(l)
            )
            btn.grid(row=row, column=col, padx=5, pady=5, ipadx=5, ipady=3)
            
            # Correct hover bindings
            btn.bind("<Enter>", lambda e, b=btn: self.on_enter(b))
            btn.bind("<Leave>", lambda e, b=btn: self.on_leave(b))

        # Back button
        back_btn = tk.Button(
            self.levels_frame,
            text="Back to Main",
            font=self.button_font,
            bg=self.colors["button_bg"],
            fg=self.colors["text"],
            activebackground=self.colors["button_active"],
            activeforeground=self.colors["text"],
            relief="flat",
            width=15,
            command=self.show_main_menu
        )
        back_btn.grid(row=2, column=0, columnspan=5, pady=20, ipadx=10, ipady=8)
        back_btn.bind("<Enter>", lambda e, b=back_btn: self.on_enter(b))
        back_btn.bind("<Leave>", lambda e, b=back_btn: self.on_leave(b))

    def on_enter(self, button):
        button.config(
            bg=self.colors["hover_bg"],
            relief="ridge",
            borderwidth=2,
            font=("Arial", button["font"].split()[1] + 2, "bold")
        )

    def on_leave(self, button):
        button.config(
            bg=self.colors["button_bg"],
            relief="flat",
            borderwidth=0,
            font=("Arial", button["font"].split()[1] - 2, "bold")
        )

    def show_main_menu(self):
        self.credits_frame.place_forget()
        self.levels_frame.place_forget()
        self.tutorial_frame.place_forget()
        self.hide_all_frames()
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")

    def show_levels(self):
        self.hide_all_frames()
        self.levels_frame.place(relx=0.5, rely=0.5, anchor="center")

    def create_credits_screen(self):
        # Credits screen elements
        self.credits_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Credits text
        credits_text = "Made by Nathan Forman, Pranaov Giridharan, Akhil Peteti, Suva Sai Ruthwik Suravarapu, DeepSeek, and ChatGPT."
        credits_label = tk.Label(
            self.credits_frame,
            text=credits_text,
            font=("Arial", 16),
            bg=self.colors["bg"],
            fg=self.colors["text"],
            wraplength=400
        )
        credits_label.pack(pady=20)
        
        # Back button
        back_btn = tk.Button(
            self.credits_frame,
            text="Back to Main",
            font=self.button_font,
            bg=self.colors["button_bg"],
            fg=self.colors["text"],
            activebackground=self.colors["button_active"],
            activeforeground=self.colors["text"],
            relief="flat",
            borderwidth=0,
            width=15,
            command=self.show_main_menu
        )
        back_btn.pack(pady=20, ipadx=10, ipady=8)
        
        # Add hover effects to back button
        back_btn.bind("<Enter>", lambda e, b=back_btn: self.on_enter(b))
        back_btn.bind("<Leave>", lambda e, b=back_btn: self.on_leave(b))

    def show_credits(self):
        self.main_frame.place_forget()
        self.credits_frame.place(relx=0.5, rely=0.5, anchor="center")

    # Modified hover effects to handle different font sizes
    def on_enter(self, button, font_size=20):
        button.config(
            bg=self.colors["hover_bg"],
            relief="ridge",
            borderwidth=2,
            font=("Arial", font_size, "bold")
        )

    def on_leave(self, button):
        button.config(
            bg=self.colors["button_bg"],
            relief="flat",
            borderwidth=0
        )
        button.config(font=("Arial", 18, "bold"))

    def show_tutorial(self): 
        self.main_frame.place_forget()
        self.tutorial_frame.place(relx=0.5, rely=0.5, anchor="center")
        

    def create_resources_screen(self):
        self.resources_frame.place(relx=.5, rely=.5, anchor="center")
    
    # Create a canvas with a scrollbar
        canvas = tk.Canvas(self.resources_frame, bg="#1a1a1a", highlightthickness=0, width=900, height=600)
        scrollbar = tk.Scrollbar(self.resources_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a1a")
        canvas.bind_all("<MouseWheel>", lambda e: self.on_mouse_wheel(e, canvas))
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind(
        "<Configure>",
        lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
        )
    )

    # Add the scrollable frame to the canvas
        canvas.create_window((0,0), window=scrollable_frame, anchor="nw")

    # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        resources_text = """
        TIPS:


Neurons: Neurons are a basic computational unit that mimic the functions of  biological neurons, that processes inputs and produces an output. Each input is associated with a weight, which determines the importance of that input. The neuron sums up the weighted inputs, adds a bias, and then applies an activation function to produce the output.
Layer: A collection of neurons that operate together at a specific depth in the network.
Input layers: 
The input layer is the first layer of a neural network. It receives the raw input data (e.g., images, text, numerical data) and passes it to the subsequent layers for processing. The input layer does not perform any computation; it simply defines the shape and size of the input data.
Convolutional layers: 
Convolutional Neural Networks are a class of deep learning models specifically designed for processing structured grid data, such as images, videos, and even audio spectrograms. CNNs are a type of artificial neural network that uses convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data. 

Flattening layers:
Flattening layers are a crucial component in Convolutional Neural Networks (CNNs) and other deep learning models. They serve as a bridge between the convolutional/pooling layers and the fully connected layers. 

Pooling layers:
The pooling layer is a critical component of Convolutional Neural Networks (CNNs) that helps reduce the spatial dimensions of the feature maps produced by convolutional layers. Its primary purpose is to downsample the feature maps, making the network computationally more efficient, reducing the risk of overfitting, and introducing a degree of translation invariance.

Fully connected layers:
Fully Connected Layers  are a fundamental component of neural networks. They play a critical role in combining features extracted by earlier layers to make predictions, such as classifying an image or predicting a value. This is where every neuron is connected to every neuron in the previous layer. It takes a flattened input and applies a linear transformation followed by a non-linear activation function.



Activation Functions: Introduce non-linearity and help the network learn complex patterns 
ReLU - ReLU is widely used because it is computationally efficient and helps mitigate the vanishing gradient problem. However, it can cause "dying ReLU" problems where some neurons become inactive.

Leaky ReLU- Leaky Rectified Linear Unit (Leaky ReLU) is an activation function used in artificial neural networks to introduce non-linearity while addressing the "dying ReLU" problem.

Softmax - typically used in the output layer for multi-class classification problems. It converts the output into a probability distribution, where each value represents the probability of a particular class.

ELU - helps to reduce the vanishing gradient problem and can lead to faster convergence compared to ReLU.

Sigmoid - The sigmoid function is a widely used activation function in neural networks, particularly in binary classification tasks. It maps any real-valued number to a value between 0 and 1, making it useful for producing probabilities.


Batch Normalization: Batch Normalization is a technique used to improve the training of neural networks by normalizing the inputs of each layer. It was introduced in a 2015 paper by Sergey Ioffe and Christian Szegedy and has since become a standard component in many deep learning models. Batch Normalization helps stabilize and accelerate training, allowing for higher learning rates and reducing the need for careful initialization of weights.Batch Normalization normalizes the outputs of a layer (or the inputs to the next layer) by adjusting and scaling the activations. Specifically, it normalizes the mean and variance of the activations for each mini-batch during training.


Dropout: Regularization technique to prevent overfitting by randomly dropping neurons during training. Prevents over-reliance on specific neurons, improving generalization.








Training a neural network:   

Forward propagation: 
Forward propagation is the process of passing input data through the network to compute predictions.  Each neuron in a layer receives inputs from the previous layer, applies a weighted sum, and passes the result through an activation function.

Loss functions:
The loss function measures how well the model’s predictions match the true labels. The goal of training is to minimize this loss by comparing the model’s predicted values to the actual values.

Mean Squared Error (MSE) Loss is a commonly used loss function in neural networks, particularly for regression tasks. It measures the average squared difference between the predicted values and the true values. The goal of training a neural network is to minimize this loss, which improves the model’s accuracy. This is the loss function used in NexusGrid.

Back propagation:
Backpropagation is the process of computing gradients of the loss function with respect to the model’s weights. These gradients are used to update the weights during training.

Weight updates:
The weights are updated iteratively using optimization algorithms such as the one described above.

Optimizations:
Optimization techniques are algorithms or methods used to adjust the parameters of a model to minimize a loss function. The goal is to find the optimal set of parameters that results in the best performance of the model.

The Adam optimizer (Adaptive Moment Estimation) is one of the most popular optimization algorithms used in training neural networks, and it's used in NexusGrid. It combines the benefits of two other optimization techniques: Momentum and RMSprop. Adam is known for its efficiency and ability to handle sparse gradients, making it a first choice for many deep learning models.
"""
# Add the text to the scrollable frame
        resources_label = tk.Label(
        scrollable_frame,
        text=resources_text,
        font=("Arial", 12),
        bg="#1a1a1a",
        fg="#ffffff",
        justify="left",
        wraplength=800
    )
        resources_label.pack(pady=20, padx=20, anchor="w")
    
    # Back button
        back_btn = tk.Button(
        scrollable_frame,
        text="Back to Main",
        font=self.button_font,
        bg=self.colors["button_bg"],
        fg=self.colors["text"],
        activebackground=self.colors["button_active"],
        activeforeground=self.colors["text"],
        relief="flat",
        borderwidth=0,
        width=15,
        command=self.show_main_menu
    )
        back_btn.pack(pady=20, ipadx=10, ipady=8)
    
    # Add hover effects to back button
        back_btn.bind("<Enter>", lambda e, b=back_btn: self.on_enter(b))
        back_btn.bind("<Leave>", lambda e, b=back_btn: self.on_leave(b))
    def show_resources(self):
        self.hide_all_frames()
        self.resources_frame.place(relx=0.5, rely=0.5, anchor="center")

if __name__ == "__main__":
    root = tk.Tk()
    app = GameLauncher(root)
    root.mainloop()