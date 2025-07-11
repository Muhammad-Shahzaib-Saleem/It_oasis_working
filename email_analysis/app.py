import streamlit as st
import pandas as pd
import os
from typing import List, Dict, Any
import time

from email_processor import EmailProcessor
from vector_db_manager import VectorDBManager
from llm_handler import LLMHandler
from config import PAGE_TITLE, PAGE_ICON, SUPPORTED_FORMATS, MAX_FILE_SIZE,VECTOR_DB_PATH

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'llm_handler' not in st.session_state:
    st.session_state.llm_handler = None
if 'emails_loaded' not in st.session_state:
    st.session_state.emails_loaded = False
if 'email_count' not in st.session_state:
    st.session_state.email_count = 0

def initialize_components():
    """Initialize vector database and LLM handler"""
    try:
        if st.session_state.vector_db is None:
            st.session_state.vector_db = VectorDBManager()
            
        
        if st.session_state.llm_handler is None:
            st.session_state.llm_handler = LLMHandler()
        
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
        return True
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return False

def process_uploaded_file(uploaded_files):
    """Process multiple uploaded email files"""
    all_valid_emails = []
    total_emails = 0

    for uploaded_file in uploaded_files:
        temp_file_path = None

        try:
            if uploaded_file is None or uploaded_file.name.strip() == "":
                continue  # Skip invalid file

            # Save uploaded file temporarily
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Determine file type
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            file_type = file_extension[1:]

            # Process emails
            processor = EmailProcessor()
            emails = processor.load_emails_from_file(temp_file_path, file_type)

            # Validate emails
            valid_emails = processor.validate_emails(emails)

            # Add to totals
            all_valid_emails.extend(valid_emails)
            total_emails += len(emails)

        except Exception as e:
            print(f"❌ Error processing file {uploaded_file.name}: {e}")

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return all_valid_emails, total_emails, len(all_valid_emails)
def get_directory_size(directory):
    total = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)  # MB


# def process_uploaded_file(uploaded_file):
#     """Process uploaded email file"""
#     temp_file_path = None
#     try:
#         # Save uploaded file temporarily
#         temp_file_path = f"temp_{uploaded_file.name}"
#         with open(temp_file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#
#         # Determine file type
#         file_extension = os.path.splitext(uploaded_file.name)[1].lower()
#         file_type = file_extension[1:]  # Remove the dot
#
#         # Process emails
#         processor = EmailProcessor()
#         emails = processor.load_emails_from_file(temp_file_path, file_type)
#
#         # Validate emails
#         valid_emails = processor.validate_emails(emails)
#
#         # Clean up temp file
#         os.remove(temp_file_path)
#
#         return valid_emails, len(emails), len(valid_emails)
#
#     except Exception as e:
#         # Clean up temp file if it exists
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)
#         raise e

def main():
    """Main application"""
    st.title("📧 Email Analysis System")
    st.markdown("Upload your email dataset and query it using natural language!")
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Database Status")
        
        # Get collection info
        if st.session_state.vector_db:
            collection_info = st.session_state.vector_db.get_collection_info()
            st.metric("Emails in Database", collection_info.get('count', 0))
            
            if collection_info.get('count', 0) > 0:
                st.success("✅ Database Ready")
                st.session_state.emails_loaded = True
                st.session_state.email_count = collection_info.get('count', 0)
            else:
                st.warning("⚠️ No emails loaded")
                st.session_state.emails_loaded = False
        
        st.markdown("---")
        
        # Database management
        st.header("🗄️ Database Management")

        if st.button("Reset All", type="primary" ,use_container_width=True):
                print("outsideeeeeeeeeeeeeeeeeeeeeeeeeeee","I m    here")
            
                if st.session_state.vector_db:
                    print("sssssssssssssssssssssssssssssssssss","I m    here")
                    if st.session_state.vector_db.delete_database():
                        print("nested iffffffff","I m    here")
                        
                        st.success("Database reset!")
                        st.session_state.emails_loaded = False
                        st.session_state.email_count = 0
                        st.rerun()
                    else:
                            
                        st.error("Failed to reset database")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Emails", "🔍 Query Emails", "📈 Analytics"])
    
    with tab1:
        st.header("Upload Email Dataset")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose your email file",
            accept_multiple_files=True,
            type=['csv', 'json', 'txt','.eml','.mbox'],
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}. Max size: {MAX_FILE_SIZE // (1024*1024)}MB",
            key=st.session_state.uploader_key
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            # Process file button
            if st.button("🚀 Process and Load Emails", type="primary"):
                try:
                    with st.spinner("Processing emails..."):
                        # Process file
                        valid_emails, total_emails, valid_count = process_uploaded_file(uploaded_files)
                        
                        # Display processing results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Emails Found", total_emails)
                        with col2:
                            st.metric("Valid Emails", valid_count)
                        with col3:
                            st.metric("Success Rate", f"{(valid_count/total_emails*100):.1f}%" if total_emails > 0 else "0%")
                        
                        if valid_count >= 10000:
                            st.success(f"✅ Found {valid_count:,} valid emails (meets minimum requirement of 1000+)")
                        elif valid_count > 0:
                            st.warning(f"⚠️ Found {valid_count} valid emails (less than 1000, but will proceed)")
                        else:
                            st.error("❌ No valid emails found in the file")
                            st.stop()
                        
                        # Add to vector database
                        with st.spinner("Creating vector database..."):
                            if st.session_state.vector_db.add_emails(valid_emails):
                                st.success(f"🎉 Successfully loaded {valid_count:,} emails into vector database!")
                                st.session_state.emails_loaded = True
                                st.session_state.email_count = valid_count
                                
                                # Show sample data
                                if valid_emails:
                                    st.subheader("📋 Sample Data Preview")
                                    sample_df = pd.DataFrame(valid_emails[:5])
                                    st.dataframe(sample_df, use_container_width=True)
                                
                                # Generate summary
                                with st.spinner("Generating dataset summary..."):
                                    summary = st.session_state.llm_handler.generate_summary(valid_emails)
                                    st.subheader("📊 Dataset Summary")
                                    st.markdown(summary)
                                
                                uploaded_files.clear()
                                st.session_state.uploader_key += 1
                                st.rerun()
                            else:
                                st.error("❌ Failed to load emails into vector database")
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.header("Query Your Emails")
        
        if not st.session_state.emails_loaded:
            st.warning("⚠️ Please upload and process emails first in the 'Upload Emails' tab.")
            st.stop()
        
        st.success(f"✅ Ready to query {st.session_state.email_count:,} emails")
        
        # Query input
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., 'Give me John's email address' or 'Find all emails from gmail.com'",
            help="Ask natural language questions about your email data"
        )
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            - **Find specific emails**: "Give me John Smith's email address"
            - **Domain search**: "Show me all emails from gmail.com"
            - **Pattern analysis**: "How many emails are from educational institutions?"
            - **Person search**: "Find emails containing 'manager' or 'director'"
            - **General search**: "Show me emails related to marketing"
            """)
        
        # Search parameters
        col1, col2 = st.columns([3, 1])
        with col2:
            num_results = st.selectbox("Results to show:", [5, 10, 15, 20], index=1)
        
        # Query button
        if st.button("🔍 Search", type="primary", disabled=not query):
            if query:
                try:
                    with st.spinner("Searching emails..."):
                        # Search vector database
                        search_results = st.session_state.vector_db.search_emails(query, num_results)
                        
                        if search_results:
                            # Generate LLM response
                            with st.spinner("Generating response..."):
                                response = st.session_state.llm_handler.generate_response(query, search_results)
                            
                            # Display response
                            st.subheader("🤖 AI Response")
                            st.markdown(response)
                            
                            # Display search results
                            st.subheader("📋 Search Results")
                            
                            for i, result in enumerate(search_results, 1):
                                with st.expander(f"Result {i} (Relevance: {(1-result['score']):.2f})"):
                                    metadata = result['metadata']
                                    
                                    # Display metadata in a nice format
                                    for key, value in metadata.items():
                                        if value and str(value).strip():
                                            st.write(f"**{key.title()}:** {value}")
                        else:
                            st.warning("No results found for your query. Try rephrasing or using different keywords.")
                
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
    
    with tab3:
        st.header("Email Analytics")
        
        if not st.session_state.emails_loaded:
            st.warning("⚠️ Please upload and process emails first to view analytics.")
            st.stop()
        
        # Get collection info
        collection_info = st.session_state.vector_db.get_collection_info()
        
        # Basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Emails", f"{collection_info.get('count', 0):,}")
        with col2:
            st.metric("Database Size", f"{get_directory_size(VECTOR_DB_PATH):.2f} MB")
        with col3:
            st.metric("Collection Name", collection_info.get('name', 'N/A'))
        
        # Quick analytics queries
        st.subheader("📊 Quick Analytics")
        
        analytics_queries = [
            "How many unique domains are in the dataset?",
            "What are the most common email domains?",
            "Analyze the email patterns in this dataset",
            "Give me statistics about this email collection"
        ]
        
        selected_query = st.selectbox("Choose an analytics query:", analytics_queries)
        
        if st.button("📈 Run Analytics", type="primary"):
            try:
                with st.spinner("Running analytics..."):
                    # Search for relevant data
                    search_results = st.session_state.vector_db.search_emails(selected_query, 20)
                    
                    # Generate analytics response
                    response = st.session_state.llm_handler.generate_response(selected_query, search_results)
                    
                    st.subheader("📊 Analytics Results")
                    st.markdown(response)
            
            except Exception as e:
                st.error(f"Error running analytics: {str(e)}")

if __name__ == "__main__":
    main()