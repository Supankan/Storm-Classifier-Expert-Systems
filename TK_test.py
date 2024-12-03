import tkinter as tk
from tkinter import ttk
from experta import *
import numpy as np
from scipy.stats import norm

class Storm(Fact):
    """Information about the storm."""
    pass

class StormExpertSystem(KnowledgeEngine):
    classifications = []
    advices = []

    categories = {
        "Mild Hurricane": {"wind_speed": (85, 5), "pressure": (970, 10), "temperature": (25, 5), "humidity": (80, 10)},
        "Moderate Hurricane": {"wind_speed": (103, 5), "pressure": (960, 10), "temperature": (25, 5), "humidity": (85, 10)},
        "Severe Hurricane": {"wind_speed": (120, 10), "pressure": (940, 10), "temperature": (25, 5), "humidity": (90, 10)},
        "Mild Thunderstorm": {"wind_speed": (60, 10), "pressure": (1000, 10), "temperature": (25, 5), "humidity": (70, 10)},
        "Moderate Thunderstorm": {"wind_speed": (50, 10), "pressure": (990, 10), "temperature": (25, 5), "humidity": (75, 10)},
        "Severe Thunderstorm": {"wind_speed": (70, 10), "pressure": (980, 10), "temperature": (25, 5), "humidity": (80, 10)},
        "Mild Winter Storm": {"wind_speed": (50, 10), "pressure": (990, 10), "temperature": (-5, 5), "humidity": (60, 10)},
        "Moderate Winter Storm": {"wind_speed": (70, 10), "pressure": (970, 10), "temperature": (-10, 5), "humidity": (65, 10)},
        "Severe Winter Storm": {"wind_speed": (90, 10), "pressure": (950, 10), "temperature": (-15, 5), "humidity": (70, 10)},
        "Calm": {"wind_speed": (20, 10), "pressure": (1010, 10), "temperature": (20, 5), "humidity": (50, 10)}
    }

    advice_map = {
        "Mild Hurricane": "Prepare for strong winds and possible flooding.",
        "Moderate Hurricane": "Expect severe damage to infrastructure.",
        "Severe Hurricane": "Evacuate immediately if in the storm's path.",
        "Mild Thunderstorm": "Stay indoors and avoid open areas.",
        "Moderate Thunderstorm": "Be cautious of lightning and heavy rain.",
        "Severe Thunderstorm": "Seek shelter immediately and avoid travel.",
        "Mild Winter Storm": "Dress warmly and avoid icy roads.",
        "Moderate Winter Storm": "Expect significant snowfall and dangerous conditions.",
        "Severe Winter Storm": "Avoid travel; power outages likely.",
        "Calm": "No action needed.",
        "High Humidity": "Stay hydrated and avoid outdoor activities.",
        "Moderate Humidity": "Be cautious of heat exhaustion.",
        "Low Humidity": "Moisturize and stay hydrated."
    }

    @Rule(Storm(wind_speed=MATCH.wind_speed, pressure=MATCH.pressure, temperature=MATCH.temperature, humidity=MATCH.humidity))
    def classify_storm(self, wind_speed, pressure, temperature, humidity):
        for category, params in self.categories.items():
            wind_log_prob = np.log(norm.pdf(wind_speed, params["wind_speed"][0], params["wind_speed"][1]) + 1e-10)
            pressure_log_prob = np.log(norm.pdf(pressure, params["pressure"][0], params["pressure"][1]) + 1e-10)
            temperature_log_prob = np.log(norm.pdf(temperature, params["temperature"][0], params["temperature"][1]) + 1e-10)
            humidity_log_prob = np.log(norm.pdf(humidity, params["humidity"][0], params["humidity"][1]) + 1e-10)
            overall_log_prob = wind_log_prob + pressure_log_prob + temperature_log_prob + humidity_log_prob
            self.classifications.append((category, overall_log_prob))
            self.advices.append((self.advice_map[category], overall_log_prob))

    @Rule(Storm(wind_speed=P(lambda x: x >= 74 and x < 96), pressure=P(lambda x: x <= 980)))
    def hurricane_mild(self):
        self.adjust_probability("Mild Hurricane", 0.8)

    @Rule(Storm(wind_speed=P(lambda x: x >= 96 and x < 111), pressure=P(lambda x: x <= 970)))
    def hurricane_moderate(self):
        self.adjust_probability("Moderate Hurricane", 0.9)

    @Rule(Storm(wind_speed=P(lambda x: x >= 111), pressure=P(lambda x: x <= 950)))
    def hurricane_severe(self):
        self.adjust_probability("Severe Hurricane", 0.95)

    @Rule(Storm(wind_speed=P(lambda x: x < 74), temperature=P(lambda x: x > 20), pressure=P(lambda x: x > 980)))
    def thunderstorm_mild(self):
        self.adjust_probability("Mild Thunderstorm", 0.7)

    @Rule(Storm(wind_speed=P(lambda x: x >= 40 and x < 60), temperature=P(lambda x: x > 20), pressure=P(lambda x: x <= 1000)))
    def thunderstorm_moderate(self):
        self.adjust_probability("Moderate Thunderstorm", 0.8)

    @Rule(Storm(wind_speed=P(lambda x: x >= 60), temperature=P(lambda x: x > 20), pressure=P(lambda x: x <= 990)))
    def thunderstorm_severe(self):
        self.adjust_probability("Severe Thunderstorm", 0.85)

    @Rule(Storm(wind_speed=P(lambda x: x >= 40 and x < 60), pressure=P(lambda x: x <= 1000), temperature=P(lambda x: x <= 0)))
    def winter_storm_mild(self):
        self.adjust_probability("Mild Winter Storm", 0.75)

    @Rule(Storm(wind_speed=P(lambda x: x >= 60 and x < 80), pressure=P(lambda x: x <= 980), temperature=P(lambda x: x <= -5)))
    def winter_storm_moderate(self):
        self.adjust_probability("Moderate Winter Storm", 0.85)

    @Rule(Storm(wind_speed=P(lambda x: x >= 80), pressure=P(lambda x: x <= 960), temperature=P(lambda x: x <= -10)))
    def winter_storm_severe(self):
        self.adjust_probability("Severe Winter Storm", 0.9)

    @Rule(Storm(wind_speed=P(lambda x: x < 30), pressure=P(lambda x: x > 1000)))
    def calm(self):
        self.adjust_probability("Calm", 1.0)

    @Rule(Storm(humidity=P(lambda x: x > 80)))
    def high_humidity(self):
        self.adjust_probability("High Humidity", 0.9)

    @Rule(Storm(humidity=P(lambda x: x >= 60 and x <= 80)))
    def moderate_humidity(self):
        self.adjust_probability("Moderate Humidity", 0.8)

    @Rule(Storm(humidity=P(lambda x: x < 60)))
    def low_humidity(self):
        self.adjust_probability("Low Humidity", 0.7)

    def adjust_probability(self, category, confidence):
        for i, (classification, log_prob) in enumerate(self.classifications):
            if classification == category:
                self.classifications[i] = (classification, log_prob + np.log(confidence))
                self.advices[i] = (self.advice_map[classification], log_prob + np.log(confidence))

    def normalize_probabilities(self):
        log_probs = [prob for _, prob in self.classifications]
        max_log_prob = max(log_probs)
        exp_probs = [np.exp(prob - max_log_prob) for prob in log_probs]
        total_exp_prob = sum(exp_probs)
        normalized_probs = [exp_prob / total_exp_prob for exp_prob in exp_probs]

        self.classifications = [(classification, normalized_probs[i]) for i, (classification, _) in enumerate(self.classifications)]
        self.advices = [(advice, normalized_probs[i]) for i, (advice, _) in enumerate(self.advices)]

def run_expert_system(wind_speed, pressure, temperature, humidity, results_frame):
    engine = StormExpertSystem()
    engine.reset()
    engine.classifications = []
    engine.advices = []
    engine.declare(Storm(wind_speed=wind_speed, pressure=pressure, temperature=temperature, humidity=humidity))
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
    for advice, probability in sorted_advices:
        if probability > 0.02:
            tk.Label(results_frame, text=f"{advice} (Probability: {probability:.4f})", fg="blue", bg="#e3e4fa", font=("Arial", 11)).pack(anchor="center", padx=10)

def main():
    root = tk.Tk()
    root.title("Storm Classification Expert System")
    root.geometry("800x800")
    root.configure(bg="#264534")

    frame = tk.Frame(root, bg="#f0f8ff", padx=20, pady=20)
    frame.pack(expand=True)

    tk.Label(frame, text="Storm Classification Expert System", font=("Arial", 20, "bold"), bg="#f0f8ff").pack()

    def update_label(value, label):
        label.config(text=f"Current Value: {float(value):.1f}")

    def create_slider(parent, label_text, from_, to_, initial_value):
        tk.Label(parent, text=label_text, font=("Arial", 10), bg="#f0f8ff").pack(anchor="w", pady=5)
        slider_frame = tk.Frame(parent, bg="#f0f8ff")
        slider_frame.pack()
        min_label = tk.Label(slider_frame, text=f"{from_}", bg="#f0f8ff")
        min_label.pack(side="left", padx=5)
        slider = ttk.Scale(slider_frame, from_=from_, to=to_, orient="horizontal", length=500)
        slider.set(initial_value)
        slider.pack(side="left", padx=5)
        max_label = tk.Label(slider_frame, text=f"{to_}", bg="#f0f8ff")
        max_label.pack(side="left", padx=5)
        value_label = tk.Label(parent, text=f"Current Value: {initial_value:.1f}", font=("Arial", 10), bg="#f0f8ff")
        value_label.pack()
        slider.bind("<Motion>", lambda event: update_label(slider.get(), value_label))
        return slider

    wind_speed_slider = create_slider(frame, "Wind Speed (mph):", 0, 150, 50)
    pressure_slider = create_slider(frame, "Pressure (hPa):", 900, 1050, 1000)
    temperature_slider = create_slider(frame, "Temperature (°C):", -20, 40, 25)
    humidity_slider = create_slider(frame, "Humidity (%):", 0, 100, 50)

    results_frame = tk.Frame(root, bg="#e3e4fa", padx=10, pady=10)
    results_frame.pack(fill="both", expand=True)

    def classify_storm():
        wind_speed = wind_speed_slider.get()
        pressure = pressure_slider.get()
        temperature = temperature_slider.get()
        humidity = humidity_slider.get()
        run_expert_system(wind_speed, pressure, temperature, humidity, results_frame)

    classify_button = tk.Button(frame, text="Classify Storm", command=classify_storm, bg="#4682b4", fg="white", font=("Arial", 12, "bold"))
    classify_button.pack(pady=15)

    root.mainloop()

if __name__ == "__main__":
    main()
