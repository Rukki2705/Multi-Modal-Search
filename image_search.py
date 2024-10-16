# image_search.py
import streamlit as st
import torch
import clip
from PIL import Image
from pinecone import Pinecone, ServerlessSpec, PineconeException
import os

def image_search_page(api_key):
    st.sidebar.title("⚙️ Image Search Options")
    
    pinecone_options = ["API Key", "Environment", "Index Name"]
    selected_pinecone_option = st.sidebar.selectbox("Pinecone Settings", pinecone_options)

    # Display Pinecone settings
    if selected_pinecone_option == "API Key":
        st.sidebar.write(f"Current API Key: {api_key}")
    elif selected_pinecone_option == "Environment":
        st.sidebar.write("Environment: us-east-1")
    elif selected_pinecone_option == "Index Name":
        st.sidebar.write("Index Name: interactive-clip-index")

    model_options = ["CLIP ViT-B/32", "Sentence-BERT"]
    selected_model_option = st.sidebar.selectbox("Models Used", model_options)

    if selected_model_option == "CLIP ViT-B/32":
        st.sidebar.write("Model: CLIP ViT-B/32 by OpenAI")
    elif selected_model_option == "Sentence-BERT":
        st.sidebar.write("Model: Sentence-BERT used for text embeddings")

    # Initialize Pinecone client
    if 'pinecone_initialized' not in st.session_state:
        try:
            pc = Pinecone(
                api_key=api_key,
                environment='us-east-1'
            )
            st.session_state.pc = pc
            st.session_state.pinecone_initialized = True
        except PineconeException as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            st.stop()

    index_name = "interactive-clip-index"

    # Create Pinecone index if not already created
    if st.button("🛠️ Create a Vector Index"):
        if 'index_created' not in st.session_state or not st.session_state.index_created:
            try:
                existing_indexes = st.session_state.pc.list_indexes()
                if index_name not in existing_indexes:
                    st.session_state.pc.create_index(
                        name=index_name,
                        dimension=512,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                st.session_state.index_created = True
                st.session_state.index = st.session_state.pc.Index(index_name)
                st.write(f"Index '{index_name}' created or connected successfully.")
            except PineconeException as e:
                st.error(f"Error creating or connecting to the index: {str(e)}")
                st.stop()
        else:
            st.write(f"Index '{index_name}' is already created.")

    # Delete button for the image index
    if st.button("❌ Delete Image Index"):
        try:
            st.write(f"Attempting to delete index '{index_name}'...")
            st.session_state.pc.delete_index(index_name)
            st.write(f"Index '{index_name}' deleted successfully.")

            # Reset session state flags
            st.session_state.index_created = False
            if 'index' in st.session_state:
                del st.session_state.index
            st.success(f"All session state related to '{index_name}' has been cleared.")
        except Exception as e:
            st.error(f"Error deleting index: {str(e)}")

    # Step 2: Upload files and preprocess images
    uploaded_files = st.file_uploader("📤 Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"{len(uploaded_files)} files uploaded successfully.")

    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 3: Preview Uploaded Images Button
    if st.button("🔍 Preview Uploaded Images"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Preview of {uploaded_file.name}", use_column_width=True)
        else:
            st.write("No files uploaded yet.")

    # Step 4: Convert images to embeddings
    if st.button("🧠 Convert to Embedding"):
        if uploaded_files:
            # Load CLIP model
            if 'model' not in st.session_state:
                model, preprocess = clip.load("ViT-B/32", device=device)
                st.session_state.model = model
                st.session_state.preprocess = preprocess
            else:
                model = st.session_state.model
                preprocess = st.session_state.preprocess

            image_directory = "images"
            os.makedirs(image_directory, exist_ok=True)

            for uploaded_file in uploaded_files:
                image_path = os.path.join(image_directory, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Preprocess the image
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

                # Generate image embedding using CLIP
                with torch.no_grad():
                    image_embedding = model.encode_image(image).cpu().numpy().flatten()

                # Store the embeddings
                if 'image_embeddings' not in st.session_state:
                    st.session_state.image_embeddings = []
                st.session_state.image_embeddings.append({"filename": uploaded_file.name, "embedding": image_embedding})

            st.write("Images converted to embeddings successfully.")
        else:
            st.write("No files uploaded yet.")

    # Step 5: Store embeddings into Pinecone only if the index exists
    if st.button("💾 Store Embeddings"):
        if 'image_embeddings' in st.session_state and 'index' in st.session_state:
            index = st.session_state.index
            for image_data in st.session_state.image_embeddings:
                index.upsert(
                    vectors=[
                        {
                            "id": image_data['filename'],
                            "values": image_data['embedding'].tolist(),
                            "metadata": {"filename": image_data['filename']}
                        }
                    ]
                )
            st.write("Embeddings stored in Pinecone successfully.")
        else:
            st.write("No embeddings to store or index not available. Please convert the images to embeddings first.")

    # Step 6: Semantic Text Search only if the index exists
    text_query = st.text_input("Enter your text query:", key="text_query_image")
    similarity_threshold = 0.20

    if st.button("🔎 Search"):
        if 'index' in st.session_state:
            index = st.session_state.index
            if text_query:
                # Ensure model is loaded for text query
                if 'model' not in st.session_state:
                    model, preprocess = clip.load("ViT-B/32", device=device)
                    st.session_state.model = model
                    st.session_state.preprocess = preprocess
                else:
                    model = st.session_state.model

                # Tokenize and generate the text embedding
                text_tokenized = clip.tokenize([text_query]).to(device)
                with torch.no_grad():
                    text_embedding = model.encode_text(text_tokenized).cpu().numpy().flatten()

                # Query Pinecone index
                query_results = index.query(
                    vector=text_embedding.tolist(),
                    top_k=2,
                    include_metadata=True
                )

                # Display results based on similarity threshold
                results_found = False
                if query_results['matches']:
                    for result in query_results['matches']:
                        if result['score'] >= similarity_threshold:
                            results_found = True
                            top_result_filename = result['metadata']['filename']
                            top_result_image_path = os.path.join("images", top_result_filename)
                            if os.path.exists(top_result_image_path):
                                top_result_image = Image.open(top_result_image_path)
                                st.image(top_result_image, caption=f"Filename: {top_result_filename} - Score: {result['score']}", use_column_width=True)
                            else:
                                st.write(f"Image file '{top_result_filename}' not found.")
                    
                    if not results_found:
                        st.write("No results found above the similarity threshold.")
                else:
                    st.write("No results found.")
        else:
            st.write("Index is not initialized. Please create an index first.")
