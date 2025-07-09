import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any
import uuid
from config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

class VectorDBManager:
    """Manage ChromaDB vector database for email storage and retrieval"""
    
    def __init__(self):
        self.db_path = VECTOR_DB_PATH
        self.collection_name = COLLECTION_NAME
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL,device='cpu')
        self.client = None
        self.collection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"Loaded existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Email data collection"}
                )
                print(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            raise Exception(f"Failed to initialize vector database: {str(e)}")
    
    def add_emails(self, emails: List[Dict[str, Any]]) -> bool:
        """Add emails to the vector database"""
        try:
            if not emails:
                return False
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for email_data in emails:
                # Use text_content for embedding
                text_content = email_data.get('text_content', '')
                if not text_content:
                    continue
                
                documents.append(text_content)
                
                # Prepare metadata (exclude text_content to avoid duplication)
                metadata = {k: str(v) for k, v in email_data.items() if k != 'text_content'}
                metadatas.append(metadata)
                
                # Generate unique ID
                email_id = str(uuid.uuid4())
                ids.append(email_id)
            
            if not documents:
                return False
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully added {len(documents)} emails to vector database")
            return True
            
        except Exception as e:
            print(f"Error adding emails to vector database: {str(e)}")
            return False
    
    def search_emails(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search emails using natural language query"""
        try:
            if not self.collection:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching emails: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            if not self.collection:
                return {}
            
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'path': self.db_path
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection"""
        try:
            if self.collection:
                # Delete the collection
                self.client.delete_collection(name=self.collection_name)
                
                # Recreate empty collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Email data collection"}
                )
                return True
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
            return False
    
    def delete_database(self) -> bool:
        """Delete the entire vector database"""
        try:
            if self.client:
                self.client.reset()
            
            # Remove database files
            import shutil
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
            
            # Reinitialize
            self._initialize_db()
            return True
        except Exception as e:
            print(f"Error deleting database: {str(e)}")
            return False