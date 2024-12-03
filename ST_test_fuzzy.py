import streamlit as st
from experta import *
import numpy as np
import skfuzzy as fuzz


# Define a Fact class for storm characteristics
class Storm(Fact):
    """Information about the storm."""
    pass


# Define the expert system class
class StormExpertSystem(KnowledgeEngine):
    classification = ""
    advice = ""
    probabilities = {}

    @Rule(Storm())
    def calculate_probabilities(self):
        # No specific crisp rule execution, as fuzzy logic determines results.
        pass


# Streamlit UI with Fuzzy Integration
def main():
    st.title("Fuzzy Storm Classification Expert System")
    st.markdown("Provide storm attributes to classify the storm with fuzzy probabilities and get safety advice.")

    # Input fields for storm attributes
    wind_speed = st.slider("Wind Speed (mph)", min_value=0, max_value=150, value=50, step=1)
    pressure = st.slider("Pressure (hPa)", min_value=900, max_value=1050, value=1000, step=1)
    temperature = st.slider("Temperature (°C)", min_value=-20, max_value=40, value=25, step=1)
    humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60, step=1)  # New humidity input

    # Fuzzy membership functions
    wind_universe = np.arange(0, 151, 1)
    pressure_universe = np.arange(900, 1051, 1)
    temp_universe = np.arange(-20, 41, 1)
    humidity_universe = np.arange(0, 101, 1)

    # Define fuzzy sets
    wind_low = fuzz.trimf(wind_universe, [0, 0, 50])
    wind_medium = fuzz.trimf(wind_universe, [30, 75, 120])
    wind_high = fuzz.trimf(wind_universe, [90, 150, 150])

    pressure_high = fuzz.trimf(pressure_universe, [1010, 1050, 1050])
    pressure_medium = fuzz.trimf(pressure_universe, [980, 1010, 1030])
    pressure_low = fuzz.trimf(pressure_universe, [900, 950, 1000])

    temp_cold = fuzz.trimf(temp_universe, [-20, -10, 0])
    temp_moderate = fuzz.trimf(temp_universe, [0, 20, 30])
    temp_hot = fuzz.trimf(temp_universe, [20, 40, 40])

    humidity_low = fuzz.trimf(humidity_universe, [0, 20, 40])
    humidity_medium = fuzz.trimf(humidity_universe, [30, 60, 90])
    humidity_high = fuzz.trimf(humidity_universe, [60, 80, 100])

    if st.button("Classify Storm"):
        # Fuzzify inputs
        wind_level_low = fuzz.interp_membership(wind_universe, wind_low, wind_speed)
        wind_level_medium = fuzz.interp_membership(wind_universe, wind_medium, wind_speed)
        wind_level_high = fuzz.interp_membership(wind_universe, wind_high, wind_speed)

        pressure_level_high = fuzz.interp_membership(pressure_universe, pressure_high, pressure)
        pressure_level_medium = fuzz.interp_membership(pressure_universe, pressure_medium, pressure)
        pressure_level_low = fuzz.interp_membership(pressure_universe, pressure_low, pressure)

        temp_level_cold = fuzz.interp_membership(temp_universe, temp_cold, temperature)
        temp_level_moderate = fuzz.interp_membership(temp_universe, temp_moderate, temperature)
        temp_level_hot = fuzz.interp_membership(temp_universe, temp_hot, temperature)

        humidity_level_low = fuzz.interp_membership(humidity_universe, humidity_low, humidity)
        humidity_level_medium = fuzz.interp_membership(humidity_universe, humidity_medium, humidity)
        humidity_level_high = fuzz.interp_membership(humidity_universe, humidity_high, humidity)

        # Calculate fuzzy probabilities for each storm type
        probabilities = {
            "Calm": min(wind_level_low, pressure_level_high, humidity_level_low),
            "Mild Thunderstorm": min(wind_level_medium, temp_level_hot, pressure_level_medium, humidity_level_medium),
            "Moderate Thunderstorm": min(wind_level_high, temp_level_hot, pressure_level_medium, humidity_level_high),
            "Mild Winter Storm": min(wind_level_medium, temp_level_cold, pressure_level_low, humidity_level_medium),
            "Moderate Hurricane": min(wind_level_high, pressure_level_low, humidity_level_high),
        }

        # Normalize probabilities
        total_sum = sum(probabilities.values())
        if total_sum > 0:
            normalized_probabilities = {k: (v / total_sum) * 100 for k, v in probabilities.items()}
        else:
            normalized_probabilities = {k: 0 for k in probabilities.keys()}

        # Sort and display results
        sorted_results = sorted(normalized_probabilities.items(), key=lambda x: x[1], reverse=True)
        st.subheader("Storm Classification Probabilities")
        for storm_type, probability in sorted_results:
            st.write(f"{storm_type}: {probability:.2f}%")

        # Display advice for the most likely classification
        most_likely = sorted_results[0][0]
        advice = {
            "Calm": "No action needed.",
            "Mild Thunderstorm": "Stay indoors and avoid open areas.",
            "Moderate Thunderstorm": "Be cautious of lightning and heavy rain.",
            "Mild Winter Storm": "Dress warmly and avoid icy roads.",
            "Moderate Hurricane": "Expect severe damage to infrastructure.",
        }
        st.subheader("Safety Advice")
        st.info(advice.get(most_likely, "Stay alert and follow safety instructions."))


if __name__ == "__main__":
    main()
