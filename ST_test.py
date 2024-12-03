import streamlit as st
from experta import *


# Define a Fact class for storm characteristics
class Storm(Fact):
    """Information about the storm."""
    pass


# Define the expert system class
class StormExpertSystem(KnowledgeEngine):
    classification = ""
    advice = ""

    # Hurricane levels
    @Rule(Storm(wind_speed=P(lambda x: x >= 74 and x < 96), pressure=P(lambda x: x <= 980)))
    def hurricane_mild(self):
        self.classification = "Mild Hurricane"
        self.advice = "Prepare for strong winds and possible flooding."

    @Rule(Storm(wind_speed=P(lambda x: x >= 96 and x < 111), pressure=P(lambda x: x <= 970)))
    def hurricane_moderate(self):
        self.classification = "Moderate Hurricane"
        self.advice = "Expect severe damage to infrastructure."

    @Rule(Storm(wind_speed=P(lambda x: x >= 111), pressure=P(lambda x: x <= 950)))
    def hurricane_severe(self):
        self.classification = "Severe Hurricane"
        self.advice = "Evacuate immediately if in the storm's path."

    # Thunderstorm levels
    @Rule(Storm(wind_speed=P(lambda x: x < 74), temperature=P(lambda x: x > 20), pressure=P(lambda x: x > 980)))
    def thunderstorm_mild(self):
        self.classification = "Mild Thunderstorm"
        self.advice = "Stay indoors and avoid open areas."

    @Rule(Storm(wind_speed=P(lambda x: x >= 40 and x < 60), temperature=P(lambda x: x > 20), pressure=P(lambda x: x <= 1000)))
    def thunderstorm_moderate(self):
        self.classification = "Moderate Thunderstorm"
        self.advice = "Be cautious of lightning and heavy rain."

    @Rule(Storm(wind_speed=P(lambda x: x >= 60), temperature=P(lambda x: x > 20), pressure=P(lambda x: x <= 990)))
    def thunderstorm_severe(self):
        self.classification = "Severe Thunderstorm"
        self.advice = "Seek shelter immediately and avoid travel."

    # Winter Storm levels
    @Rule(Storm(wind_speed=P(lambda x: x >= 40 and x < 60), pressure=P(lambda x: x <= 1000), temperature=P(lambda x: x <= 0)))
    def winter_storm_mild(self):
        self.classification = "Mild Winter Storm"
        self.advice = "Dress warmly and avoid icy roads."

    @Rule(Storm(wind_speed=P(lambda x: x >= 60 and x < 80), pressure=P(lambda x: x <= 980), temperature=P(lambda x: x <= -5)))
    def winter_storm_moderate(self):
        self.classification = "Moderate Winter Storm"
        self.advice = "Expect significant snowfall and dangerous conditions."

    @Rule(Storm(wind_speed=P(lambda x: x >= 80), pressure=P(lambda x: x <= 960), temperature=P(lambda x: x <= -10)))
    def winter_storm_severe(self):
        self.classification = "Severe Winter Storm"
        self.advice = "Avoid travel; power outages likely."

    # Calm condition
    @Rule(Storm(wind_speed=P(lambda x: x < 30), pressure=P(lambda x: x > 1000)))
    def calm(self):
        self.classification = "Calm"
        self.advice = "No action needed."


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

        # Display the classification and advice
        if engine.classification:
            st.subheader("Storm Classification")
            st.success(engine.classification)
            st.subheader("Safety Advice")
            st.info(engine.advice)
        else:
            st.warning("No classification matched. Adjust the inputs and try again.")


if __name__ == "__main__":
    main()
