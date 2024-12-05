from experta import *
import numpy as np
from scipy.stats import norm
from geopy.distance import geodesic

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

    @Rule(Storm(wind_speed=MATCH.wind_speed, pressure=MATCH.pressure, temperature=MATCH.temperature, humidity=MATCH.humidity, storm_location=MATCH.storm_location, user_location=MATCH.user_location))
    def classify_storm(self, wind_speed, pressure, temperature, humidity, storm_location, user_location):
        for category, params in self.categories.items():
            wind_log_prob = np.log(norm.pdf(wind_speed, params["wind_speed"][0], params["wind_speed"][1]) + 1e-10)
            pressure_log_prob = np.log(norm.pdf(pressure, params["pressure"][0], params["pressure"][1]) + 1e-10)
            temperature_log_prob = np.log(norm.pdf(temperature, params["temperature"][0], params["temperature"][1]) + 1e-10)
            humidity_log_prob = np.log(norm.pdf(humidity, params["humidity"][0], params["humidity"][1]) + 1e-10)
            overall_log_prob = wind_log_prob + pressure_log_prob + temperature_log_prob + humidity_log_prob
            self.classifications.append((category, overall_log_prob))
            self.advices.append((self.advice_map[category], overall_log_prob))

        # Calculate distance between storm location and user location
        distance = geodesic(storm_location, user_location).kilometers
        self.advices.append((f"Distance to storm: {distance:.2f} km", 1.0))

        # Add advice based on distance
        if distance < 50:
            self.advices.append(("Move away immediately!", 1.0))
        elif distance < 100:
            self.advices.append(("Prepare to evacuate.", 1.0))
        elif distance < 200:
            self.advices.append(("Stay alert and monitor the situation.", 1.0))
        else:
            self.advices.append(("You are safe for now.", 1.0))


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

    @Rule(Storm(wind_speed=P(lambda x: x >= 40 and x < 60), temperature=P(lambda x: x > 20),
                pressure=P(lambda x: x <= 1000)))
    def thunderstorm_moderate(self):
        self.adjust_probability("Moderate Thunderstorm", 0.8)

    @Rule(Storm(wind_speed=P(lambda x: x >= 60), temperature=P(lambda x: x > 20), pressure=P(lambda x: x <= 990)))
    def thunderstorm_severe(self):
        self.adjust_probability("Severe Thunderstorm", 0.85)

    @Rule(Storm(wind_speed=P(lambda x: x >= 40 and x < 60), pressure=P(lambda x: x <= 1000),
                temperature=P(lambda x: x <= 0)))
    def winter_storm_mild(self):
        self.adjust_probability("Mild Winter Storm", 0.75)

    @Rule(Storm(wind_speed=P(lambda x: x >= 60 and x < 80), pressure=P(lambda x: x <= 980),
                temperature=P(lambda x: x <= -5)))
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
                self.advices[i] = (self.advice_map[category], log_prob + np.log(confidence))

    def normalize_probabilities(self):
        log_probs = [prob for _, prob in self.classifications]
        max_log_prob = max(log_probs)
        exp_probs = [np.exp(prob - max_log_prob) for prob in log_probs]
        total_exp_prob = sum(exp_probs)
        normalized_probs = [exp_prob / total_exp_prob for exp_prob in exp_probs]

        self.classifications = [(classification, normalized_probs[i]) for i, (classification, _) in enumerate(self.classifications)]

        # Ensure advices list is updated correctly
        advice_probs = [prob for _, prob in self.advices]
        max_advice_prob = max(advice_probs)
        exp_advice_probs = [np.exp(prob - max_advice_prob) for prob in advice_probs]
        total_exp_advice_prob = sum(exp_advice_probs)
        normalized_advice_probs = [exp_prob / total_exp_advice_prob for exp_prob in exp_advice_probs]

        self.advices = [(advice, normalized_advice_probs[i]) for i, (advice, _) in enumerate(self.advices)]
