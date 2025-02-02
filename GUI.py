import tkinter as tk
from tkinter import font
import winsound
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter.simpledialog as simpledialog  # Import simpledialog for user input
import funcLibrary

class TutorialLevel:
    def __init__(self, master, back_callback):
        self.master = master
        self.back_callback = back_callback
        self.dragging = None
        
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
        ax.plot(x, np.sin(x), color="#3498db")
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
        # Assume the header is in row 0 and actual components are in rows > 0.
        layer_items = []
        for widget in self.palette_frame.grid_slaves():
            # grid_slaves() may return widgets in reverse order;
            info = widget.grid_info()
            if int(info["row"]) == 0:
                continue  # Skip header
            if hasattr(widget, "model_data"):
                layer_items.append((int(info["row"]), widget.model_data))
        # Sort layers in order of increasing row number
        layer_items.sort(key=lambda x: x[0])
        # Extract only the model_data part into a list
        input_list = [layer for _, layer in layer_items]

        # Create the neural network model using your makemodel() function.
        # (Make sure to import makemodel from its module at the top of your file.)
        model = funcLibrary.makemodel(input_list)
        print("Neural network created:")
        print(model)

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
        
        self.create_main_menu()
        self.create_credits_screen()
        self.create_levels_screen()
        self.create_tutorial_screen()
        
        self.show_main_menu()
        
    def show_tutorial(self):
        self.hide_all_frames()
        self.tutorial_frame.place(relx=0.5, rely=0.5, anchor="center")

    def create_tutorial_screen(self):
        self.tutorial = TutorialLevel(self.tutorial_frame, self.show_main_menu)

    def hide_all_frames(self):
        self.main_frame.place_forget()
        self.credits_frame.place_forget()
        self.levels_frame.place_forget()
        self.tutorial_frame.place_forget()

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

    def show_levels(self):
        self.main_frame.place_forget()
        self.credits_frame.place_forget()
        self.levels_frame.place(relx=0.5, rely=0.5, anchor="center")

    def create_credits_screen(self):
        # Credits screen elements
        self.credits_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Credits text
        credits_text = "Made by Nathan Forman, Pranaov Giridharan, and DeepSeek."
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