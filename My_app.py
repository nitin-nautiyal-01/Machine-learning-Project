import streamlit as st

def main():
    st.title("Simple Form with Streamlit")

    # Input fields
    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # Submit button
    if st.button("Submit"):
        # Display submitted data
        st.success(f"Name: {name}\nEmail: {email}\nPassword: {password}")

if __name__ == "__main__":
    main()
