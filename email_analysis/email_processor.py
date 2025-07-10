import pandas as pd
import json
import re
from typing import List, Dict, Any
from email_validator import validate_email, EmailNotValidError
import email
from bs4 import BeautifulSoup
import os
from email.utils import parseaddr
import mailbox
class EmailProcessor:
    """Process and validate email data from various file formats"""
    
    def __init__(self):
        self.processed_emails = []
    
    def load_emails_from_file(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Load emails from different file formats"""
        try:
            if file_type == 'csv':
                return self._load_from_csv(file_path)
            elif file_type == 'json':
                return self._load_from_json(file_path)
            elif file_type == 'txt':
                return self._load_from_txt(file_path)
            elif file_type == 'eml':    
                return self._load_from_eml(file_path)
            elif file_type == 'mbox':
                return self._load_from_mbox(file_path)            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise Exception(f"Error loading emails: {str(e)}")
    
    def _load_from_csv(self, file_path: str) -> List[Dict[str, Any]]:
    
        """Load emails from CSV file"""
        df = pd.read_csv(file_path)
        
        # Try to identify email columns automatically
        email_columns = self._identify_email_columns(df)
        
        emails = []
        for _, row in df.iterrows():
            email_data = {}
            for col in df.columns:
                if col in email_columns:
                    email_data['email'] = row[col]
                else:
                    email_data[col] = row[col]
            
            # Create a text representation for vector search
            email_data['text_content'] = self._create_text_content(email_data)
            emails.append(email_data)
            print(emails)
        
        return emails
    
    def _load_from_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load emails from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            emails = data
        elif isinstance(data, dict):
            emails = [data]
        else:
            raise ValueError("Invalid JSON format")
        
        # Process each email
        for email in emails:
            email['text_content'] = self._create_text_content(email)
        
        return emails

    def _load_from_eml(self, file_path: str) -> List[Dict[str, Any]]:
        """Load emails from .eml file or folder of .eml files"""
        eml_files = []

        if os.path.isdir(file_path):
            eml_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.eml')]
        elif file_path.endswith('.eml'):
            eml_files = [file_path]
        else:
            raise ValueError("Path must be a .eml file or a directory containing .eml files")

        emails = [self._parse_eml(fp) for fp in eml_files]
        
        # Create text content for vector DB
        for email_data in emails:
            email_data["text_content"] = self._create_text_content(email_data)

        return emails

    def _parse_eml(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_file(f)

        body = ""
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body += part.get_payload(decode=True).decode(errors='ignore')
            elif content_type == "text/html":
                soup = BeautifulSoup(part.get_payload(decode=True).decode(errors='ignore'), 'html.parser')
                body += soup.get_text()
        
        _, from_email = parseaddr(msg.get("from", ""))
        _, to_email = parseaddr(msg.get("to", ""))
        
        return {
            "subject": msg.get("subject", ""),
            "from": from_email,
            "to": to_email,
            "date": msg.get("date", ""),
            "body": body.strip()
        }
    def _load_from_mbox(self, file_path: str) -> List[Dict[str, Any]]:
        """Load emails from mbox file"""
        emails = []
        try:
            mbox = mailbox.mbox(file_path)
            emails = [self._parse_and_enrich(message) for message in mbox]
        except Exception as e:
            print(f"Error parsing message: {e}")
        finally:
            mbox.close()
        return emails
    
    def _parse_and_enrich(self, message):
        parsed = self._parse_mbox(message)
        parsed["text_content"] = self._create_text_content(parsed)
        return parsed

    def _parse_mbox(self, msg: email.message.Message) -> Dict[str, Any]:
        """Parses an email message 
        into a dictionary
        with subject, from, to, date, and body."""
        body = ""
        # if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body += part.get_payload(decode=True).decode(errors='ignore')
            elif content_type == "text/html":
                soup = BeautifulSoup(part.get_payload(decode=True).decode(errors='ignore'), 'html.parser')
                body += soup.get_text()
        
        _, from_email = parseaddr(msg.get("from", ""))
        _, to_email = parseaddr(msg.get("to", ""))
    
        return {
        "subject": msg.get("subject", ""),
        "from": from_email,
        "to": to_email,
        "date": msg.get("date", ""),
        "body": body.strip()
    }

       
    
    def _load_from_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Load emails from text file (one email per line or structured format)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to extract emails using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails_found = re.findall(email_pattern, content)
        
        emails = []
        for email in emails_found:
            email_data = {
                'email': email,
                'text_content': email
            }
            emails.append(email_data)
        
        return emails
    
    def _identify_email_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that likely contain email addresses"""
        email_columns = []
        
        for col in df.columns:
            # Check column name
            if any(keyword in col.lower() for keyword in ['email', 'mail', 'e-mail']):
                email_columns.append(col)
                continue
            
            # Check content pattern
            sample_values = df[col].dropna().head(10)
            email_count = 0
            for value in sample_values:
                if self._is_valid_email(str(value)):
                    email_count += 1
            
            if email_count > len(sample_values) * 0.5:  # More than 50% are emails
                email_columns.append(col)
        
        return email_columns
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email address"""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    def _create_text_content(self, email_data: Dict[str, Any]) -> str:
        """Create searchable text content from email data"""
        text_parts = []
        
        for key, value in email_data.items():
            if key != 'text_content' and value is not None:
                text_parts.append(f"{key}: {str(value)}")
        
        return " | ".join(text_parts)
    
    def validate_emails(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean email data"""
        valid_emails = []
        
        for email_data in emails:
            email_valid = 'email' in email_data and self._is_valid_email(email_data['email'])
            from_to_valid = (
            'from' in email_data and
            'to' in email_data and
            self._is_valid_email(email_data['from']) and
            self._is_valid_email(email_data['to'])
        )
            if email_valid or from_to_valid:
                valid_emails.append(email_data)
        return valid_emails