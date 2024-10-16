# text_search.py
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec, PineconeException
import textwrap

def text_search_page(api_key):
    st.sidebar.title("⚙️ Text Search Options")
    
    pinecone_options = ["API Key", "Environment", "Index Name"]
    selected_pinecone_option = st.sidebar.selectbox("Pinecone Settings", pinecone_options)

    # Display Pinecone settings
    if selected_pinecone_option == "API Key":
        st.sidebar.write(f"Current API Key: {api_key}")
    elif selected_pinecone_option == "Environment":
        st.sidebar.write("Environment: us-east-1")
    elif selected_pinecone_option == "Index Name":
        st.sidebar.write("Index Name: sentence-transformers-pdf-index")

    # Model Dropdown in Sidebar for Text
    model_options = ["CLIP ViT-B/32", "Sentence-BERT"]
    selected_model_option = st.sidebar.selectbox("Models Used", model_options)

    if selected_model_option == "Sentence-BERT":
        st.sidebar.write("Model: Sentence-BERT for Text Search")
    elif selected_model_option == "CLIP ViT-B/32":
        st.sidebar.write("Model: CLIP ViT-B/32 by OpenAI")

    # Initialize Pinecone client for text
    if 'pinecone_initialized_text' not in st.session_state:
        try:
            pc_text = Pinecone(
                api_key=api_key,
                environment='us-east-1'
            )
            st.session_state.pc_text = pc_text
            st.session_state.pinecone_initialized_text = True
        except PineconeException as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            st.stop()

    # Button to create the index for text search
    index_name = "sentence-transformers-pdf-index"
    if st.button("🛠️ Create a Vector Index"):
        if 'index_created_text' not in st.session_state or not st.session_state.index_created_text:
            try:
                existing_indexes = st.session_state.pc_text.list_indexes()
                if index_name not in existing_indexes:
                    st.write(f"Creating a new index '{index_name}'...")
                    st.session_state.pc_text.create_index(
                        name=index_name,
                        dimension=384,  # Sentence-BERT outputs 384-dimensional embeddings
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                st.session_state.index_created_text = True
                st.session_state.index_text = st.session_state.pc_text.Index(index_name)
                st.write(f"Index '{index_name}' created or connected successfully.")
            except PineconeException as e:
                st.error(f"Error creating or connecting to the index: {str(e)}")
                st.stop()
        else:
            st.write(f"Index '{index_name}' is already created.")

    # Delete button for the text index
    if st.button("❌ Delete Text Index"):
        try:
            st.write(f"Attempting to delete index '{index_name}'...")
            st.session_state.pc_text.delete_index(index_name)
            st.write(f"Index '{index_name}' deleted successfully.")

            # Reset session state flags
            st.session_state.index_created_text = False
            if 'index_text' in st.session_state:
                del st.session_state.index_text
            st.success(f"All session state related to '{index_name}' has been cleared.")
        except Exception as e:
            st.error(f"Error deleting index: {str(e)}")

    # Connect to the index if it exists
    if 'index_created_text' in st.session_state and st.session_state.index_created_text:
        if 'index_text' not in st.session_state:
            try:
                st.session_state.index_text = st.session_state.pc_text.Index(index_name)
                st.write("Connected to the existing index.")
            except PineconeException as e:
                st.error(f"Error connecting to the index: {str(e)}")

    # Step 2: Upload the PDF file
    uploaded_pdf = st.file_uploader("📤 Choose a PDF file", type="pdf")

    if uploaded_pdf:
        st.write(f"File uploaded: {uploaded_pdf.name}")

    # Step 3: Preview PDF content
    if uploaded_pdf and st.button("🔍 Preview PDF"):
        try:
            reader = PdfReader(uploaded_pdf)
            pages = [page.extract_text() for page in reader.pages]

            for i, page in enumerate(pages):
                if page:
                    st.write(f"### Page {i + 1}")
                    wrapped_page = textwrap.fill(page, width=100)
                    st.write(wrapped_page)
                else:
                    st.write(f"### Page {i + 1} is empty or could not be read.")
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")

    # Step 4: Convert PDF to embeddings
    if st.button("🧠 Convert PDF to Embedding"):
        if uploaded_pdf:
            try:
                reader = PdfReader(uploaded_pdf)
                pages = [page.extract_text() for page in reader.pages]
                model = SentenceTransformer('all-MiniLM-L6-v2')

                # Store embeddings in session state before storing in Pinecone
                st.session_state.text_embeddings = []
                for idx, page_text in enumerate(pages):
                    if page_text:
                        page_embedding = model.encode(page_text)
                        st.session_state.text_embeddings.append({"page_id": f"page_{idx}", "embedding": page_embedding, "content": page_text})
                st.write("PDF converted to embeddings successfully.")
            except Exception as e:
                st.error(f"Error converting PDF to embeddings: {str(e)}")
        else:
            st.write("Please upload a PDF file first.")

    # Step 5: Store embeddings in Pinecone
    if st.button("💾 Store Embeddings"):
        if 'text_embeddings' in st.session_state and 'index_text' in st.session_state:
            index = st.session_state.index_text
            for page_data in st.session_state.text_embeddings:
                index.upsert(
                    vectors=[
                        {
                            "id": page_data['page_id'],
                            "values": page_data['embedding'].tolist(),
                            "metadata": {"page_content": page_data['content']}
                        }
                    ]
                )
            st.write("Embeddings stored in Pinecone successfully.")
        else:
            st.write("No embeddings to store or index not available. Please convert the PDF to embeddings first.")

    # Step 6: Perform text-based search
    text_query = st.text_input("Enter your search query:")
    similarity_threshold = 0.20
    context_characters = 200

    if st.button("🔎 Search PDF"):
        if 'index_text' in st.session_state:
            index = st.session_state.index_text
            if text_query:
                try:
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    query_embedding = model.encode(text_query)

                    query_results = index.query(
                        vector=query_embedding.tolist(),
                        top_k=2,
                        include_metadata=True
                    )

                    results_found = False
                    if query_results['matches']:
                        for result in query_results['matches']:
                            if result['score'] >= similarity_threshold:
                                page_content = result['metadata']['page_content'].replace("\n", " ")
                                page_number = int(result['id'].split('_')[-1]) + 1

                                query_position = page_content.lower().find(text_query.lower())
                                if query_position != -1:
                                    results_found = True
                                    start = max(query_position - context_characters, 0)
                                    end = min(query_position + len(text_query) + context_characters, len(page_content))
                                    displayed_content = page_content[start:end]
                                    highlighted_content = displayed_content.replace(text_query, f"**{text_query}**")
                                    if start > 0:
                                        highlighted_content = "..." + highlighted_content
                                    if end < len(page_content):
                                        highlighted_content += "..."
                                    st.write(f"### Page {page_number}")
                                    st.write(f"Matched Page Content:\n{'-' * 40}\n{highlighted_content}\n{'-' * 40}")
                                    st.write(f"Score: {result['score']}\n")
                                else:
                                    st.write(f"### Page {page_number}")
                                    st.write("Query not found in page content.")
                                    st.write(f"Score: {result['score']}\n")
                        if not results_found:
                            st.write("No results found above the similarity threshold.")
                    else:
                        st.write("No results found.")
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
        else:
            st.write("Index is not initialized. Please create an index first.")
