import tkinter as tk
from tkinter import font
import winsound

class GameLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Main Menu")
        self.root.geometry("800x600")
        self.root.configure(bg="#1a1a1a")
        self.root.minsize(600, 400)
        
        # Grayscale color scheme
        self.colors = {
            "bg": "#1a1a1a",
            "button_bg": "#4d4d4d",
            "button_active": "#666666",
            "text": "#ffffff",
            "hover_bg": "#666666"
        }
        
        # Create all frames but don't place them yet
        self.main_frame = tk.Frame(root, bg=self.colors["bg"])
        self.credits_frame = tk.Frame(root, bg=self.colors["bg"])
        self.levels_frame = tk.Frame(root, bg=self.colors["bg"])
        
        # Initialize all screens
        self.create_main_menu()
        self.create_credits_screen()
        self.create_levels_screen()
        
        # Show only main menu at startup
        self.show_main_menu()

    def create_main_menu(self):
        self.button_font = font.Font(family="Arial", size=18, weight="bold")
        
        self.buttons = [
            ("Levels", self.show_levels),
            ("Tutorial", self.show_tutorial),
            ("Settings", self.open_settings),
            ("Credits", self.show_credits)
        ]
        
        for idx, (text, command) in enumerate(self.buttons):
            btn = tk.Button(
                self.main_frame,
                text=text,
                font=self.button_font,
                bg=self.colors["button_bg"],
                fg=self.colors["text"],
                activebackground=self.colors["button_active"],
                activeforeground=self.colors["text"],
                relief="flat",
                borderwidth=0,
                width=15,
                command=command
            )
            btn.grid(row=idx, column=0, pady=10, ipadx=10, ipady=8)
            
            # Fixed hover bindings
            btn.bind("<Enter>", lambda e, b=btn: self.on_enter(b))
            btn.bind("<Leave>", lambda e, b=btn: self.on_leave(b))

    def create_levels_screen(self):
        # Level buttons grid
        for level in range(1, 11):
            row = (level-1) // 5
            col = (level-1) % 5
            
            btn = tk.Button(
                self.levels_frame,
                text=f"Level {level}",
                font=("Arial", 14),
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
        self.levels_frame.place_forget()
        self.credits_frame.place_forget()
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

    def show_main_menu(self):
        self.credits_frame.place_forget()
        self.levels_frame.place_forget()
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")

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

    # Placeholder methods for other buttons
    def show_tutorial(self): pass
    def open_settings(self): pass

if __name__ == "__main__":
    root = tk.Tk()
    app = GameLauncher(root)
    root.mainloop()