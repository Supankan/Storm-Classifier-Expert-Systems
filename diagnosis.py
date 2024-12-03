from experta import *


class Symptom(Fact):
    """Fact for storing symptoms"""
    pass


class DiagnosisSystem(KnowledgeEngine):
    @Rule(Symptom(fever=True) & Symptom(headeache=True) & Symptom(cough=True))
    def flu(self):
        self.declare(Fact(diagnosis="Flu", explanation="Fever, headache, and cough suggest flu"))

    @Rule(Symptom(cough=True) & NOT(Symptom(fever=True)))
    def common_cold(self):
        self.declare(Fact(diagnosis="Common Cold", explanation="Cough without fever suggest common cold"))

    @Rule(AS.fact << Symptom())
    def unknown(self):
        self.declare(Fact(diagnosis="Unknown", explanation="Unable to determine diagnosis based on symptom"))


def diagnose(symptoms):
    """Run the reasoning engine with the provided symptoms."""
    engine = DiagnosisSystem()
    engine.reset()  # Reset the engine state
    for symptom, value in symptoms.items():
        engine.declare(Symptom(**{symptom: value}))
    engine.run()

    # Extract results
    results = [fact for fact in engine.facts.values() if 'diagnosis' in fact]
    return results

