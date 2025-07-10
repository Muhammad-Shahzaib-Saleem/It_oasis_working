#!/usr/bin/env python3
"""
Test script to verify all components work correctly
"""

import os
import sys
from email_processor import EmailProcessor
from vector_db_manager import VectorDBManager

def test_email_processor():
    """Test email processing functionality"""
    print("Testing Email Processor...")
    
    processor = EmailProcessor()
    
    # Test CSV loading
    try:
        emails = processor.load_emails_from_file('sample_emails.csv', 'csv')
        print(f"✅ Loaded {len(emails)} emails from CSV")
        
        # Validate emails
        valid_emails = processor.validate_emails(emails)
        print(f"✅ Validated {len(valid_emails)} emails")
        
        return valid_emails[:10]  # Return first 10 for testing
        
    except Exception as e:
        print(f"❌ Email processor test failed: {e}")
        return []

def test_vector_db():
    """Test vector database functionality"""
    print("\nTesting Vector Database...")
    
    try:
        # Initialize vector DB
        vector_db = VectorDBManager()
        print("✅ Vector database initialized")
        
        # Clear any existing data
        vector_db.clear_collection()
        print("✅ Database cleared")
        
        return vector_db
        
    except Exception as e:
        print(f"❌ Vector database test failed: {e}")
        return None

def test_integration(emails, vector_db):
    """Test integration between components"""
    print("\nTesting Integration...")
    
    if not emails or not vector_db:
        print("❌ Cannot test integration - missing components")
        return
    
    try:
        # Add emails to vector database
        success = vector_db.add_emails(emails)
        if success:
            print("✅ Emails added to vector database")
        else:
            print("❌ Failed to add emails to vector database")
            return
        
        # Test search
        results = vector_db.search_emails("gmail", 3)
        print(f"✅ Search returned {len(results)} results")
        
        # Show sample result
        if results:
            print(f"Sample result: {results[0]['metadata'].get('email', 'N/A')}")
        
        # Get collection info
        info = vector_db.get_collection_info()
        print(f"✅ Collection contains {info.get('count', 0)} emails")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

def main():
    """Run all tests"""
    print("🧪 Running Component Tests\n")
    
    # Test email processor
    emails = test_email_processor()
    
    # Test vector database
    vector_db = test_vector_db()
    
    # Test integration
    test_integration(emails, vector_db)
    
    print("\n🎉 Tests completed!")
    print("\nTo run the full application:")
    print("1. Copy .env.example to .env")
    print("2. Add your GROQ_API_KEY to .env")
    print("3. Run: streamlit run app.py")

if __name__ == "__main__":
    main()