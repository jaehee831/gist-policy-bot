import openai
import streamlit as st

openai.api_key = 'api key'

def generate_response(prompt):
    completions = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1024,
        stop=None,
        temperature=0,
        top_p=1,
    )
    message = completions["choices"][0]["text"].replace("\n", "")
    return message

# Custom CSS to include in the header for background color and other styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Adding the banner image
st.image(r"..\gist.jpg", use_column_width=True)

st.header("🤖 Gist Policy App (Demo)")

# Image uploader
image = st.file_uploader("Upload an Image", type=["jpg", "png"])

if image is not None:
    st.image(image, caption='Uploaded Image.', use_column_width=True)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('You: ', '', key='input')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.write(f'**You:** {st.session_state["past"][i]}')
        st.write(f'**Bot:** {st.session_state["generated"][i]}')
