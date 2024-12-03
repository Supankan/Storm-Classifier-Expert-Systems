import streamlit as st
from diagnosis import diagnose  # Import the expert system logic

# Page title
st.title("Medical Diagnosis Expert System")
st.write("This expert system provides a diagnosis based on your symptoms.")

# Create checkboxes for symptoms
st.subheader("Select your symptoms:")
symptoms = {
    'fever': st.checkbox('Fever'),
    'headache': st.checkbox('Headache'),
    'cough': st.checkbox('Cough')
}

# Diagnose button
if st.button('Diagnose'):
    results = diagnose(symptoms)
    if not results:
        st.error("No diagnosis found.")
    else:
        for result in results:
            st.subheader(f"Diagnosis: {result['diagnosis']}")
            st.write(result['explanation'])
