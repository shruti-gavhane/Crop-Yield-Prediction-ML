import tkinter as tk
from tkinter import messagebox, font

# Function to calculate yield
def calculate_yield():
    try:
        year = int(entry_year.get())
        dist_name = entry_dist.get()
        crop = entry_crop.get()
        area = float(entry_area.get())  # in 1000 ha
        production = float(entry_production.get())  # in 1000 tons
        rainfall = float(entry_rain.get())
        temp = float(entry_temp.get())

        yield_kg_per_ha = (production * 1000) / (area * 1000)
        yield_label.config(text=f"Yield (Kg/ha): {yield_kg_per_ha:.2f}", fg="#00FFAA")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

# GUI Setup
root = tk.Tk()
root.title("🌾 Crop Yield Calculator")
root.geometry("450x700")
root.configure(bg="#202A44")  # Dark blue background

# Fonts
title_font = ("Helvetica", 20, "bold")
label_font = ("Calibri", 12)
entry_font = ("Calibri", 12)

# Title
tk.Label(root, text="Crop Yield Predictor", bg="#202A44", fg="#F9DC5C", font=title_font).pack(pady=20)

# Helper function to create styled label and entry
def create_field(label, var):
    tk.Label(root, text=label, bg="#202A44", fg="white", font=label_font).pack()
    e = tk.Entry(root, textvariable=var, font=entry_font, width=30, bg="#F5F5F5")
    e.pack(pady=5)
    return e

# Input fields
entry_year_var = tk.StringVar()
entry_dist_var = tk.StringVar()
entry_crop_var = tk.StringVar()
entry_area_var = tk.StringVar()
entry_production_var = tk.StringVar()
entry_rain_var = tk.StringVar()
entry_temp_var = tk.StringVar()

entry_year = create_field("📅 Year", entry_year_var)
entry_dist = create_field("📍 District Name", entry_dist_var)
entry_crop = create_field("🌿 Crop", entry_crop_var)
entry_area = create_field("🌾 Area (1000 ha)", entry_area_var)
entry_production = create_field("🏭 Production (1000 tons)", entry_production_var)
entry_rain = create_field("🌧️ Total Rainfall (mm)", entry_rain_var)
entry_temp = create_field("🌡️ Avg Temperature (°C)", entry_temp_var)

# Button
tk.Button(root, text="✨ Calculate Yield", command=calculate_yield, font=("Arial", 12, "bold"),
          bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=20)

# Output
yield_label = tk.Label(root, text="Yield (Kg/ha): --", font=("Courier", 16, "bold"),
                       bg="#202A44", fg="#FFA726")
yield_label.pack(pady=15)

# Run GUI
root.mainloop()
