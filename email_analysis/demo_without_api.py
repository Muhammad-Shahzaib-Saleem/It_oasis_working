#!/usr/bin/env python3
"""
Demo script to show email analysis functionality without requiring Groq API
"""

import pandas as pd
from email_processor import EmailProcessor
from vector_db_manager import VectorDBManager

def demo_email_analysis():
    """Demonstrate the email analysis system"""
    print("🎯 Email Analysis System Demo")
    print("="*50)
    
    # Initialize components
    print("1. Initializing components...")
    processor = EmailProcessor()
    vector_db = VectorDBManager()
    
    # Clear existing data
    vector_db.clear_collection()
    
    # Load sample emails
    print("2. Loading sample emails...")
    emails = processor.load_emails_from_file('sample_emails.csv', 'csv')
    valid_emails = processor.validate_emails(emails)
    print(f"   📧 Loaded {len(valid_emails)} valid emails")
    
    # Add to vector database
    print("3. Creating vector database...")
    success = vector_db.add_emails(valid_emails)
    if success:
        print("   ✅ Vector database created successfully")
    else:
        print("   ❌ Failed to create vector database")
        return
    
    # Show database info
    info = vector_db.get_collection_info()
    print(f"   📊 Database contains {info.get('count', 0)} emails")
    
    # Demonstrate search functionality
    print("\n4. Demonstrating search functionality...")
    
    search_queries = [
        "gmail",
        "john",
        "manager",
        "company.com",
        "tech"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Searching for: '{query}'")
        results = vector_db.search_emails(query, 3)
        
        if results:
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                email = metadata.get('email', 'N/A')
                name = metadata.get('full_name', 'N/A')
                company = metadata.get('company', 'N/A')
                relevance = 1 - result['distance']
                
                print(f"   {i}. {email} | {name} | {company} (Relevance: {relevance:.2f})")
        else:
            print("   No results found")
    
    # Show analytics
    print("\n5. Basic Analytics...")
    
    # Domain analysis
    domains = {}
    for email in valid_emails:
        if 'email' in email:
            domain = email['email'].split('@')[-1]
            domains[domain] = domains.get(domain, 0) + 1
    
    print("   📈 Top email domains:")
    sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)
    for domain, count in sorted_domains[:5]:
        print(f"      {domain}: {count} emails")
    
    # Company analysis
    companies = {}
    for email in valid_emails:
        if 'company' in email:
            company = email['company']
            companies[company] = companies.get(company, 0) + 1
    
    print("\n   🏢 Top companies:")
    sorted_companies = sorted(companies.items(), key=lambda x: x[1], reverse=True)
    for company, count in sorted_companies[:5]:
        print(f"      {company}: {count} employees")
    
    print("\n6. Sample Data Preview...")
    df = pd.DataFrame(valid_emails[:5])
    print(df[['full_name', 'email', 'company', 'job_title']].to_string(index=False))
    
    print("\n🎉 Demo completed!")
    print("\nTo use the full system with AI responses:")
    print("1. Get a Groq API key from https://console.groq.com/")
    print("2. Create .env file with GROQ_API_KEY=your_key")
    print("3. Run: python run_app.py")

if __name__ == "__main__":
    demo_email_analysis()