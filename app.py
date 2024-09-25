import streamlit as st
from utils import chatbot, text
from streamlit_chat import message

def main():

    st.set_page_config(page_title='ChatBot PDF', page_icon=':books:')

    st.header('Converse com seus arquivos')
    user_question = st.text_input('Faça uma pergunta para mim:')

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if user_question and st.session_state.conversation:

        response = st.session_state.conversation(user_question)
        print(response)

        chat_history = response.get('chat_history', [])

        if chat_history:
            answer = chat_history[-1].content

            st.session_state.chat_history.append({'message': user_question, 'is_user': True})
            st.session_state.chat_history.append({'message': answer, 'is_user': False})

    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            message(chat['message'], is_user=chat['is_user'])

    with st.sidebar:

        st.subheader('Seus Arquivos')
        pdf_docs = st.file_uploader('Carregue os seus arquivos', accept_multiple_files=True)

        if st.button('Processar'):
            try:
                all_files_text = text.process_files(pdf_docs)
                print(all_files_text)

                chunks, total_tokens = text.create_text_chunks(all_files_text)
                max_tokens = 16000

                if total_tokens > max_tokens:
                    st.warning(f"O número de tokens ({total_tokens}) excede o limite de {max_tokens} tokens.")
                    if st.button('OK'):
                        st.experimental_rerun()
                else:
                    vectorstore = chatbot.create_vectorstore(chunks)
                    print(vectorstore)

                    st.session_state.conversation = chatbot.create_conversation_chain(vectorstore)

                    st.success(f"Processamento feito com sucesso! Número de tokens: {total_tokens}")
                    if st.button('OK'):
                        pass  # Você pode adicionar alguma ação aqui, se desejar

            except Exception as e:
                st.error("Ocorreu um erro durante o processamento.")
                if st.button('OK'):
                    st.experimental_rerun()
                # O erro não será mostrado na aplicação

if __name__ == '__main__':

    main()
