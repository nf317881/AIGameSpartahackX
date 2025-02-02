import math
import tkinter as tk
from tkinter import font
import winsound
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter.simpledialog as simpledialog  # Import simpledialog for user input
import funcLibrary

model_list = []

class Level:
    def __init__(self, master, back_callback, score, classification, func = None, MNIST_images = None):
        self.master = master
        self.back_callback = back_callback
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
        # Remove network builder...
        self.create_toolbox()
        self.create_palette()
        self.create_control_buttons()

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
        self.palette_frame = tk.Frame(self.frame, bg="#34495e", width=400, height=400)
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
                    if type(params) == list:
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

        if not self.classification:
            x_train, x_test, y_train, y_test = funcLibrary.generate_data(self.func)
        else:
            x_train, x_test, y_train, y_test = funcLibrary.get_custom_mnist(self.MNIST_images)


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
        # Add your logic for loading next level here
        print("Proceeding to next level...")


    def reset_network(self):
        # Reset the palette: destroy all children except the header (assumed at row 0)
        for widget in self.palette_frame.grid_slaves():
            if int(widget.grid_info()["row"]) != 0:
                widget.destroy()

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

        self.level_frames = []
        for x in range(10):
            self.level_frames.append(tk.Frame(root, bg="#1a1a1a"))
        
        self.create_main_menu()
        self.create_credits_screen()
        self.create_levels_screen()
        self.create_tutorial_screen()
        
        self.show_main_menu()
        self.create_each_level()
        
    def show_tutorial(self):
        self.hide_all_frames()
        self.tutorial_frame.place(relx=0.5, rely=0.5, anchor="center")

    def create_each_level(self):
        self.levels = []
        self.levels.append(Level(self.level_frames[0], self.show_levels, 3E8, False, func = lambda x: abs(x))) #1
        self.levels.append(Level(self.level_frames[1], self.show_levels, 3E8, False, func = lambda x: math.sqrt(x))) #2
        self.levels.append(Level(self.level_frames[2], self.show_levels, 3E8, False, func = lambda x: math.log(x+6))) #3
        self.levels.append(Level(self.level_frames[3], self.show_levels, 3E8, False, func = lambda x: x**2)) #4
        self.levels.append(Level(self.level_frames[4], self.show_levels, 3E8, False, func = lambda x: math.sin(x))) #5
        self.levels.append(Level(self.level_frames[5], self.show_levels, 3E8, False, func = lambda x: abs(x))) #6
        self.levels.append(Level(self.level_frames[6], self.show_levels, 3E8, False, func = lambda x: abs(x))) #7
        self.levels.append(Level(self.level_frames[7], self.show_levels, 3E8, False, func = lambda x: abs(x))) #8
        self.levels.append(Level(self.level_frames[8], self.show_levels, 3E8, False, func = lambda x: abs(x))) #9
        self.levels.append(Level(self.level_frames[9], self.show_levels, 3E8, False, func = lambda x: abs(x))) #10

    def create_tutorial_screen(self):
        self.tutorial = Level(self.tutorial_frame, self.show_main_menu, 3E8, False, func = lambda x: x)

    def hide_all_frames(self):
        self.main_frame.place_forget()
        self.credits_frame.place_forget()
        self.levels_frame.place_forget()
        self.tutorial_frame.place_forget()
        for x in range(10):
            self.level_frames[x].place_forget()

    def create_main_menu(self):
        self.button_font = font.Font(family="Arial", size=18, weight="bold")
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")
        for idx, (text, command) in enumerate([
                ("Levels", self.show_levels),
                ("Tutorial", self.show_tutorial),
                ("Settings", self.open_settings),
                ("Credits", self.show_credits)]):
            btn = tk.Button(self.main_frame, text=text, font=self.button_font, 
                            bg="#4d4d4d", fg="#ffffff", activebackground="#666666", 
                            activeforeground="#ffffff", relief="flat", borderwidth=0, 
                            width=15, command=command)
            btn.grid(row=idx, column=0, pady=10, ipadx=10, ipady=8)

    def create_levels_screen(self):
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

    def start_level(self, num):
        self.hide_all_frames()
        self.level_frames[num-1].place(relx=0.5, rely=0.5, anchor="center")

    def show_levels(self):
        self.main_frame.place_forget()
        self.credits_frame.place_forget()
        self.levels_frame.place(relx=0.5, rely=0.5, anchor="center")

    def create_credits_screen(self):
        # Credits screen elements
        self.credits_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Credits text
        credits_text = "Made by Nathan Forman, Pranaov Giridharan, Akhil, Suva, and DeepSeek."
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
        back_btn.bind("<Button-1>", self.play_sound)

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

    def play_sound(self, event):
        winsound.Beep(440, 0)

    def show_tutorial(self): 
        self.main_frame.place_forget()
        self.tutorial_frame.place(relx=0.5, rely=0.5, anchor="center")
        

    def open_settings(self): pass

if __name__ == "__main__":
    root = tk.Tk()
    app = GameLauncher(root)
    root.mainloop()