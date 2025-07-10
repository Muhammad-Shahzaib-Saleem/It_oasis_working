import json
import csv
import random
from faker import Faker

fake = Faker()

def generate_sample_emails(count=1000):
    """Generate sample email data for testing"""
    
    # Common domains
    domains = [
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'company.com', 'business.org', 'startup.io', 'tech.net',
        'university.edu', 'school.edu', 'government.gov', 'nonprofit.org'
    ]
    
    # Job titles
    job_titles = [
        'Software Engineer', 'Product Manager', 'Data Scientist', 'Designer',
        'Marketing Manager', 'Sales Representative', 'CEO', 'CTO', 'CFO',
        'HR Manager', 'Operations Manager', 'Business Analyst', 'Consultant',
        'Project Manager', 'Developer', 'Architect', 'Director', 'VP'
    ]
    
    # Companies
    companies = [
        'TechCorp', 'InnovateLab', 'DataSystems', 'CloudWorks', 'AI Solutions',
        'Digital Dynamics', 'Future Tech', 'Smart Systems', 'Global Corp',
        'Startup Inc', 'Enterprise Ltd', 'Innovation Hub', 'Tech Pioneers'
    ]
    
    emails = []
    
    for i in range(count):
        # Generate person data
        first_name = fake.first_name()
        last_name = fake.last_name()
        
        # Generate email
        domain = random.choice(domains)
        email_formats = [
            f"{first_name.lower()}.{last_name.lower()}@{domain}",
            f"{first_name.lower()}{last_name.lower()}@{domain}",
            f"{first_name[0].lower()}{last_name.lower()}@{domain}",
            f"{first_name.lower()}{random.randint(1, 999)}@{domain}"
        ]
        email = random.choice(email_formats)
        
        # Generate other data
        company = random.choice(companies)
        job_title = random.choice(job_titles)
        phone = fake.phone_number()
        city = fake.city()
        country = fake.country()
        
        email_data = {
            'first_name': first_name,
            'last_name': last_name,
            'full_name': f"{first_name} {last_name}",
            'email': email,
            'company': company,
            'job_title': job_title,
            'phone': phone,
            'city': city,
            'country': country,
            'domain': domain
        }
        
        emails.append(email_data)
    
    return emails

def save_sample_data():
    """Generate and save sample data in different formats"""
    
    print("Generating sample email data...")
    emails = generate_sample_emails(1200)  # Generate more than 1000
    
    # Save as CSV
    print("Saving CSV file...")
    with open('sample_emails.csv', 'w', newline='', encoding='utf-8') as f:
        if emails:
            writer = csv.DictWriter(f, fieldnames=emails[0].keys())
            writer.writeheader()
            writer.writerows(emails)
    
    # Save as JSON
    print("Saving JSON file...")
    with open('sample_emails.json', 'w', encoding='utf-8') as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)
    
    # Save as TXT (just emails)
    print("Saving TXT file...")
    with open('sample_emails.txt', 'w', encoding='utf-8') as f:
        for email_data in emails:
            f.write(f"{email_data['email']}\n")
    
    print(f"Generated {len(emails)} sample emails in 3 formats:")
    print("- sample_emails.csv")
    print("- sample_emails.json") 
    print("- sample_emails.txt")

if __name__ == "__main__":
    # Install faker if not available
    try:
        from faker import Faker
    except ImportError:
        print("Installing faker...")
        import subprocess
        subprocess.check_call(["pip", "install", "faker"])
        from faker import Faker
    
    save_sample_data()