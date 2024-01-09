import streamlit as st

def main():
    st.set_page_config(page_title="Form Submission", page_icon="✅")

    st.title("Simple Form with Streamlit")

    # Input fields
    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # Submit button
    if st.button("Submit"):
        # Display submitted data on a new page
        st.write(f"## Form Submitted Successfully!")
        st.write(f"**Name:** {name}")
        st.write(f"**Email:** {email}")
        st.write(f"**Password:** {password}")

if __name__ == "__main__":
    main()
