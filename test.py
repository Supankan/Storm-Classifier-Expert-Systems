from experta import *

# Define a Fact class for storm characteristics
class Storm(Fact):
    """Information about the storm."""
    pass

# Define the expert system class
class StormExpertSystem(KnowledgeEngine):
    # Hurricane levels
    @Rule(Storm(wind_speed=P(lambda x: x >= 74 and x < 96), pressure=P(lambda x: x <= 980)))
    def hurricane_mild(self):
        print("This storm is classified as a Mild Hurricane. Prepare for strong winds and possible flooding.")

    @Rule(Storm(wind_speed=P(lambda x: x >= 96 and x < 111), pressure=P(lambda x: x <= 970)))
    def hurricane_moderate(self):
        print("This storm is classified as a Moderate Hurricane. Expect severe damage to infrastructure.")

    @Rule(Storm(wind_speed=P(lambda x: x >= 111), pressure=P(lambda x: x <= 950)))
    def hurricane_severe(self):
        print("This storm is classified as a Severe Hurricane. Evacuate immediately if in the storm's path.")

    # Thunderstorm levels
    @Rule(Storm(wind_speed=P(lambda x: x < 74), temperature=P(lambda x: x > 20), pressure=P(lambda x: x > 980)))
    def thunderstorm_mild(self):
        print("This storm is classified as a Mild Thunderstorm. Stay indoors and avoid open areas.")

    @Rule(Storm(wind_speed=P(lambda x: x >= 40 and x < 60), temperature=P(lambda x: x > 20), pressure=P(lambda x: x <= 1000)))
    def thunderstorm_moderate(self):
        print("This storm is classified as a Moderate Thunderstorm. Be cautious of lightning and heavy rain.")

    @Rule(Storm(wind_speed=P(lambda x: x >= 60), temperature=P(lambda x: x > 20), pressure=P(lambda x: x <= 990)))
    def thunderstorm_severe(self):
        print("This storm is classified as a Severe Thunderstorm. Seek shelter immediately and avoid travel.")

    # Winter Storm levels
    @Rule(Storm(wind_speed=P(lambda x: x >= 40 and x < 60), pressure=P(lambda x: x <= 1000), temperature=P(lambda x: x <= 0)))
    def winter_storm_mild(self):
        print("This storm is classified as a Mild Winter Storm. Dress warmly and avoid icy roads.")

    @Rule(Storm(wind_speed=P(lambda x: x >= 60 and x < 80), pressure=P(lambda x: x <= 980), temperature=P(lambda x: x <= -5)))
    def winter_storm_moderate(self):
        print("This storm is classified as a Moderate Winter Storm. Expect significant snowfall and dangerous conditions.")

    @Rule(Storm(wind_speed=P(lambda x: x >= 80), pressure=P(lambda x: x <= 960), temperature=P(lambda x: x <= -10)))
    def winter_storm_severe(self):
        print("This storm is classified as a Severe Winter Storm. Avoid travel; power outages likely.")

    # Calm condition
    @Rule(Storm(wind_speed=P(lambda x: x < 30), pressure=P(lambda x: x > 1000)))
    def calm(self):
        print("This is not a storm; conditions are calm. No action needed.")

# Main function to run the expert system
if __name__ == "__main__":
    # Initialize the expert system
    engine = StormExpertSystem()

    # Reset the knowledge engine
    engine.reset()

    # Declare facts about different storms
    print("Storm Analysis:")
    engine.declare(Storm(wind_speed=90, pressure=965, temperature=28))   # Moderate Hurricane
    engine.declare(Storm(wind_speed=25, pressure=1015, temperature=12))  # Calm
    engine.declare(Storm(wind_speed=70, pressure=985, temperature=-5))   # Severe Winter Storm
    engine.declare(Storm(wind_speed=50, pressure=995, temperature=25))   # Moderate Thunderstorm
    engine.declare(Storm(wind_speed=120, pressure=940, temperature=30))  # Severe Hurricane

    # Run the expert system
    engine.run()
