import streamlit as st
from experta import *
import numpy as np
from scipy.stats import norm


# Define a Fact class for storm characteristics
class Storm(Fact):
    """Information about the storm."""
    pass


# Define the expert system class
class StormExpertSystem(KnowledgeEngine):
    classifications = []
    advices = []

    # Define mean and standard deviation for each category
    categories = {
        "Mild Hurricane": {"wind_speed": (85, 5), "pressure": (970, 10), "temperature": (25, 5)},
        "Moderate Hurricane": {"wind_speed": (103, 5), "pressure": (960, 10), "temperature": (25, 5)},
        "Severe Hurricane": {"wind_speed": (120, 10), "pressure": (940, 10), "temperature": (25, 5)},
        "Mild Thunderstorm": {"wind_speed": (60, 10), "pressure": (1000, 10), "temperature": (25, 5)},
        "Moderate Thunderstorm": {"wind_speed": (50, 10), "pressure": (990, 10), "temperature": (25, 5)},
        "Severe Thunderstorm": {"wind_speed": (70, 10), "pressure": (980, 10), "temperature": (25, 5)},
        "Mild Winter Storm": {"wind_speed": (50, 10), "pressure": (990, 10), "temperature": (-5, 5)},
        "Moderate Winter Storm": {"wind_speed": (70, 10), "pressure": (970, 10), "temperature": (-10, 5)},
        "Severe Winter Storm": {"wind_speed": (90, 10), "pressure": (950, 10), "temperature": (-15, 5)},
        "Calm": {"wind_speed": (20, 10), "pressure": (1010, 10), "temperature": (20, 5)}
    }

    # Define advice for each category
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
        "Calm": "No action needed."
    }

    @Rule(Storm(wind_speed=MATCH.wind_speed, pressure=MATCH.pressure, temperature=MATCH.temperature))
    def classify_storm(self, wind_speed, pressure, temperature):
        for category, params in self.categories.items():
            wind_log_prob = np.log(norm.pdf(wind_speed, params["wind_speed"][0], params["wind_speed"][1]) + 1e-10)
            pressure_log_prob = np.log(norm.pdf(pressure, params["pressure"][0], params["pressure"][1]) + 1e-10)
            temperature_log_prob = np.log(norm.pdf(temperature, params["temperature"][0], params["temperature"][1]) + 1e-10)
            overall_log_prob = wind_log_prob + pressure_log_prob + temperature_log_prob
            self.classifications.append((category, overall_log_prob))
            self.advices.append((self.advice_map[category], overall_log_prob))

    # Hurricane levels
    @Rule(Storm(wind_speed=P(lambda x: x >= 74 and x < 96), pressure=P(lambda x: x <= 980)))
    def hurricane_mild(self):
        self.adjust_probability("Mild Hurricane", 0.8)

    @Rule(Storm(wind_speed=P(lambda x: x >= 96 and x < 111), pressure=P(lambda x: x <= 970)))
    def hurricane_moderate(self):
        self.adjust_probability("Moderate Hurricane", 0.9)

    @Rule(Storm(wind_speed=P(lambda x: x >= 111), pressure=P(lambda x: x <= 950)))
    def hurricane_severe(self):
        self.adjust_probability("Severe Hurricane", 0.95)

    # Thunderstorm levels
    @Rule(Storm(wind_speed=P(lambda x: x < 74), temperature=P(lambda x: x > 20), pressure=P(lambda x: x > 980)))
    def thunderstorm_mild(self):
        self.adjust_probability("Mild Thunderstorm", 0.7)

    @Rule(Storm(wind_speed=P(lambda x: x >= 40 and x < 60), temperature=P(lambda x: x > 20), pressure=P(lambda x: x <= 1000)))
    def thunderstorm_moderate(self):
        self.adjust_probability("Moderate Thunderstorm", 0.8)

    @Rule(Storm(wind_speed=P(lambda x: x >= 60), temperature=P(lambda x: x > 20), pressure=P(lambda x: x <= 990)))
    def thunderstorm_severe(self):
        self.adjust_probability("Severe Thunderstorm", 0.85)

    # Winter Storm levels
    @Rule(Storm(wind_speed=P(lambda x: x >= 40 and x < 60), pressure=P(lambda x: x <= 1000), temperature=P(lambda x: x <= 0)))
    def winter_storm_mild(self):
        self.adjust_probability("Mild Winter Storm", 0.75)

    @Rule(Storm(wind_speed=P(lambda x: x >= 60 and x < 80), pressure=P(lambda x: x <= 980), temperature=P(lambda x: x <= -5)))
    def winter_storm_moderate(self):
        self.adjust_probability("Moderate Winter Storm", 0.85)

    @Rule(Storm(wind_speed=P(lambda x: x >= 80), pressure=P(lambda x: x <= 960), temperature=P(lambda x: x <= -10)))
    def winter_storm_severe(self):
        self.adjust_probability("Severe Winter Storm", 0.9)

    # Calm condition
    @Rule(Storm(wind_speed=P(lambda x: x < 30), pressure=P(lambda x: x > 1000)))
    def calm(self):
        self.adjust_probability("Calm", 1.0)

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


# Streamlit UI
def main():
    st.title("Storm Classification Expert System")
    st.markdown("Provide storm attributes to classify the storm and get safety advice.")

    # Input fields for storm attributes
    wind_speed = st.slider("Wind Speed (mph)", min_value=0, max_value=150, value=50, step=1)
    pressure = st.slider("Pressure (hPa)", min_value=900, max_value=1050, value=1000, step=1)
    temperature = st.slider("Temperature (°C)", min_value=-20, max_value=40, value=25, step=1)

    # Run the expert system when the user clicks the button
    if st.button("Classify Storm"):
        # Initialize the expert system
        engine = StormExpertSystem()
        engine.reset()

        # Declare the storm attributes
        engine.declare(Storm(wind_speed=wind_speed, pressure=pressure, temperature=temperature))

        # Run the inference engine
        engine.run()

        # Normalize the probabilities
        engine.normalize_probabilities()

        # Sort classifications and advice by probability in ascending order
        sorted_classifications = sorted(engine.classifications, key=lambda x: x[1], reverse=True)
        sorted_advices = sorted(engine.advices, key=lambda x: x[1], reverse=True)

        # Display the classifications and advice
        if sorted_classifications:
            st.subheader("Storm Classifications (Ascending Probability)")
            for classification, probability in sorted_classifications:
                if probability > 0.02:  # Only display non-zero probability
                    st.success(f"{classification} (Probability: {probability:.4f})")
            st.subheader("Safety Advice (Ascending Probability)")
            for advice, probability in sorted_advices:
                if probability > 0.02:  # Only display non-zero probability
                    st.info(f"{advice} (Probability: {probability:.4f})")
        else:
            st.warning("No classification matched. Adjust the inputs and try again.")


if __name__ == "__main__":
    main()
