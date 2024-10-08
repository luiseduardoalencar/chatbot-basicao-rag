# ChatBot PDF

Este é um aplicativo Streamlit que permite que você converse com seus arquivos PDF. Você pode fazer perguntas sobre o conteúdo dos PDFs carregados, e o chatbot responderá com base nas informações dos arquivos.

## Funcionalidades

- **Converse com PDFs**: Carregue múltiplos arquivos PDF e interaja com o conteúdo deles através de um chatbot inteligente.
- **Processamento de Texto**: O aplicativo processa o texto dos PDFs e cria uma cadeia de conversação usando modelos de linguagem avançados.
- **Interação em Tempo Real**: Faça perguntas e receba respostas instantaneamente.
- **Mensagens Informativas**: Após o processamento, uma caixa de diálogo aparece informando "Processamento feito com sucesso" e exibe a quantidade de tokens utilizados.
- **Limite de Tokens**: Se o número de tokens exceder o limite estabelecido (por exemplo, 16.000 tokens), uma mensagem de erro é exibida através de uma caixa de diálogo. Ao clicar em "OK", o erro não aparece mais na aplicação.
- **Gerenciamento de Erros**: Erros são tratados e exibidos de forma amigável, sem expor detalhes técnicos ao usuário.

## Pré-requisitos

- **Python 3.10** ou superior
- **Conta na OpenAI** com uma chave de API válida
- **Poetry** instalado para gerenciamento de dependências

## Instalação

1. **Clone este repositório:**

   ```bash
   git clone https://github.com/seu-usuario/chatbot-pdf.git
   cd chatbot-pdf
   ```

2. **Instale o Poetry (se ainda não o tiver):**

   ```bash
   pipx install poetry
   ```

   Certifique-se de que o caminho do Poetry está no seu `PATH`. Reinicie o terminal ou siga as instruções exibidas após a instalação.

3. **Instale as dependências com o Poetry:**

   ```bash
   poetry install
   ```

4. **Ative o ambiente virtual do Poetry:**

   ```bash
   poetry shell
   ```

5. **Configure a chave de API da OpenAI:**

   Crie um arquivo `.env` na raiz do projeto e adicione sua chave de API:

   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

   Substitua `your-openai-api-key` pela sua chave real.

## Uso

1. **Execute o aplicativo Streamlit:**

   ```bash
   poetry -vvv streamlit run app.py
   ```

2. **Interaja com o ChatBot:**

   - No navegador que se abrir, carregue seus arquivos PDF usando o menu lateral.
   - Clique no botão **"Processar"**.
   - Após o processamento (uma mensagem de sucesso será exibida mostrando o número de tokens), faça perguntas no campo de texto principal.

## Estrutura do Projeto

- `app.py`: Arquivo principal que executa o aplicativo Streamlit.
- `chatbot.py`: Contém funções para criar o vectorstore e a cadeia de conversação.
- `text.py`: Contém funções para processar os arquivos PDF e dividir o texto em chunks.
- `utils/`: Diretório que contém os módulos `chatbot.py` e `text.py`.
- `pyproject.toml`: Arquivo de configuração do Poetry que lista as dependências do projeto.
- `.env`: Arquivo que contém variáveis de ambiente (como a chave da API).
- `README.md`: Este arquivo, contendo as instruções e informações do projeto.

## Dependências

As dependências são gerenciadas pelo Poetry e estão listadas no arquivo `pyproject.toml`. As principais dependências incluem:

- **streamlit**: Framework web para construir aplicações interativas em Python.
- **python-dotenv**: Carrega variáveis de ambiente de um arquivo `.env`.
- **langchain** e **langchain-community**: Ferramentas para trabalhar com modelos de linguagem e cadeias de conversação.
- **openai**: Biblioteca oficial da OpenAI para acesso aos modelos GPT.
- **PyPDF2**: Biblioteca para manipulação de arquivos PDF.
- **faiss-cpu**: Biblioteca para busca vetorial eficiente.
- **tiktoken**: Biblioteca para tokenização compatível com modelos da OpenAI.
- **streamlit-chat**: Componente para adicionar recursos de chat ao Streamlit.

## Configuração do Poetry

Seu arquivo `pyproject.toml` deve se parecer com:

```toml
[tool.poetry]
name = "chatbot-pdf"
version = "0.1.0"
description = "Chat PDF"
authors = ["Luis Eduardo <luiseduardoalencarmelo@gmail.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
streamlit = "^1.38.0"
python-dotenv = "^1.0.1"
langchain = "^0.0.308"
langchain-community = "^0.0.7"
openai = "^0.28.1"
pypdf2 = "^3.0.1"
faiss-cpu = "^1.8.0.post1"
tiktoken = "^0.7.0"
streamlit-chat = "^0.1.1"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

## Contribuição

Sinta-se à vontade para contribuir com este projeto. Faça um fork, crie uma branch e abra um Pull Request.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Autor

- **Luis Eduardo** - [luiseduardoalencarmelo@gmail.com](mailto:luiseduardoalencarmelo@gmail.com)

---

Se tiver alguma dúvida ou precisar de assistência, por favor, não hesite em entrar em contato.
