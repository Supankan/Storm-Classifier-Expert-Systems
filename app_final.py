import tkinter as tk
from tkinter import ttk
from rules_final import Storm, StormExpertSystem
import random
from geopy.distance import geodesic

def run_expert_system(wind_speed, pressure, temperature, humidity, storm_location, user_location, results_frame):
    engine = StormExpertSystem()
    engine.reset()
    engine.classifications = []
    engine.advices = []
    engine.declare(Storm(wind_speed=wind_speed, pressure=pressure, temperature=temperature, humidity=humidity, storm_location=storm_location, user_location=user_location))
    engine.run()
    engine.normalize_probabilities()

    sorted_classifications = sorted(engine.classifications, key=lambda x: x[1], reverse=True)
    sorted_advices = sorted(engine.advices, key=lambda x: x[1], reverse=True)

    for widget in results_frame.winfo_children():
        widget.destroy()

    tk.Label(results_frame, text="Storm Classifications", font=("Arial", 14, "bold"), bg="#e3e4fa").pack(pady=5)
    for classification, probability in sorted_classifications:
        if probability > 0.02:
            tk.Label(results_frame, text=f"{classification} (Probability: {probability:.4f})", fg="green", bg="#e3e4fa", font=("Arial", 11)).pack(anchor="center", padx=10)

    tk.Label(results_frame, text="Safety Advice", font=("Arial", 14, "bold"), bg="#e3e4fa").pack(pady=5)
    if storm_location is None or user_location is None:
        tk.Label(results_frame, text="Please enter the coordinates for both storm and user locations.", fg="red", bg="#e3e4fa", font=("Arial", 11)).pack(anchor="center", padx=10)
    else:
        for advice, _ in sorted_advices:
            if "Distance to storm" in advice or "Move away immediately!" in advice or "Prepare to evacuate." in advice or "Stay alert and monitor the situation." in advice or "You are safe for now." in advice:
                tk.Label(results_frame, text=f"{advice}", fg="blue", bg="#e3e4fa", font=("Arial", 11)).pack(anchor="center", padx=10)

    tk.Label(results_frame, text="Additional Advice", font=("Arial", 14, "bold"), bg="#e3e4fa").pack(pady=5)
    for classification, probability in sorted_classifications:
        if probability > 0.02:
            advice = engine.advice_map.get(classification, "No additional advice available.")
            tk.Label(results_frame, text=f"{advice} (Probability: {probability:.4f})", fg="purple", bg="#e3e4fa", font=("Arial", 11)).pack(anchor="center", padx=10)

def main():
    root = tk.Tk()
    root.title("Extreme Weather Expert System")
    root.geometry("1000x800")
    root.configure(bg="#264534")

    frame = tk.Frame(root, bg="#f0f8ff", padx=20, pady=20)
    frame.pack(expand=True)

    tk.Label(frame, text="Storm Classification Expert System", font=("Arial", 20, "bold"), bg="#f0f8ff").pack()

    def create_entry(parent, label_text, initial_value):
        entry_frame = tk.Frame(parent, bg="#f0f8ff")
        entry_frame.pack(anchor="w", pady=5, fill="x")
        tk.Label(entry_frame, text=label_text, font=("Arial", 12), bg="#f0f8ff").pack(side="left", padx=5)
        entry = tk.Entry(entry_frame, font=("Arial", 12), bg="#f0f8ff")
        entry.insert(0, initial_value)
        entry.pack(side="left", padx=5, fill="x", expand=True)
        return entry

    wind_speed_entry = create_entry(frame, "Wind Speed (mph):", "")
    pressure_entry = create_entry(frame, "Pressure (hPa):", "")

    def create_slider(parent, label_text, from_, to_, initial_value):
        tk.Label(parent, text=label_text, font=("Arial", 12), bg="#f0f8ff").pack(anchor="w", pady=5)
        slider_frame = tk.Frame(parent, bg="#f0f8ff")
        slider_frame.pack()
        slider = ttk.Scale(slider_frame, from_=from_, to=to_, orient="horizontal", length=500)
        slider.set(initial_value)
        slider.pack()
        value_label = tk.Label(parent, text=f"Current Value: {initial_value:.1f}", font=("Arial", 12), bg="#f0f8ff")
        value_label.pack()
        slider.bind("<Motion>", lambda event: update_label(slider.get(), value_label))
        return slider

    def update_label(value, label):
        label.config(text=f"Current Value: {float(value):.1f}")

    temperature_slider = create_slider(frame, "Temperature (°C):", -20, 40, 20)
    humidity_slider = create_slider(frame, "Humidity (%):", 0, 100, 50)

    tk.Label(frame, text="Storm Location (latitude, longitude):", font=("Arial", 12), bg="#f0f8ff").pack(anchor="w", pady=5)
    storm_location_frame = tk.Frame(frame, bg="#f0f8ff")
    storm_location_frame.pack(anchor="w", pady=5)
    tk.Label(storm_location_frame, text="e.g., (34.0522, -118.2437):", font=("Arial", 12), bg="#f0f8ff").pack(side="left", padx=5)
    storm_location_entry = tk.Entry(storm_location_frame, font=("Arial", 12), bg="#f0f8ff")
    storm_location_entry.pack(side="left", padx=5)

    tk.Label(frame, text="User Location (latitude, longitude):", font=("Arial", 12), bg="#f0f8ff").pack(anchor="w", pady=5)
    user_location_frame = tk.Frame(frame, bg="#f0f8ff")
    user_location_frame.pack(anchor="w", pady=5)
    tk.Label(user_location_frame, text="e.g., (34.0522, -118.2437):", font=("Arial", 12), bg="#f0f8ff").pack(side="left", padx=5)
    user_location_entry = tk.Entry(user_location_frame, font=("Arial", 12), bg="#f0f8ff")
    user_location_entry.pack(side="left", padx=5)

    results_frame = tk.Frame(frame, bg="#e3e4fa", padx=20, pady=20)
    results_frame.pack(pady=20)

    def get_random_value(min_val, max_val):
        return random.randint(min_val, max_val)

    def get_random_location():
        return (random.uniform(-90, 90), random.uniform(-180, 180))

    def parse_location(entry):
        try:
            lat, lon = map(float, entry.split(','))
            return (lat, lon)
        except ValueError:
            return None

    def estimate_missing_values(temperature, humidity):
        # Example heuristics for estimating missing values
        if temperature is None and humidity is None:
            temperature = 20  # Default temperature
            humidity = 50  # Default humidity
        elif temperature is None:
            if humidity > 70:
                temperature = 25  # High humidity typically means warmer temperatures
            elif humidity < 30:
                temperature = 10  # Low humidity typically means cooler temperatures
            else:
                temperature = 20  # Default temperature
        elif humidity is None:
            if temperature > 30:
                humidity = 70  # Warmer temperatures typically mean higher humidity
            elif temperature < 10:
                humidity = 30  # Cooler temperatures typically mean lower humidity
            else:
                humidity = 50  # Default humidity
        else:
            if temperature > 30 and humidity is None:
                humidity = 70  # Warmer temperatures typically mean higher humidity
            elif temperature < 10 and humidity is None:
                humidity = 30  # Cooler temperatures typically mean lower humidity
            elif humidity > 70 and temperature is None:
                temperature = 25  # High humidity typically means warmer temperatures
            elif humidity < 30 and temperature is None:
                temperature = 10  # Low humidity typically means cooler temperatures

        return temperature, humidity

    run_button = tk.Button(
        frame,
        text="Classify",
        command=lambda: run_expert_system(
            int(wind_speed_entry.get()) if wind_speed_entry.get() else get_random_value(0, 150),
            int(pressure_entry.get()) if pressure_entry.get() else get_random_value(900, 1050),
            temperature_slider.get() if temperature_slider.get() else None,
            humidity_slider.get() if humidity_slider.get() else None,
            parse_location(storm_location_entry.get()) if storm_location_entry.get() else None,
            parse_location(user_location_entry.get()) if user_location_entry.get() else None,
            results_frame,
        ),
        font=("Arial", 12, "bold"),
        bg="#4CAF50",
        fg="white",
        activebackground="#45a049",
        activeforeground="white",
        relief="raised",
        bd=3,
        padx=10,
        pady=5,
    )
    run_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
