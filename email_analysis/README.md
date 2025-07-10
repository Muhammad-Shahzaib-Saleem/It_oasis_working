# Email Analysis System

A powerful email analysis system that converts email datasets into a vector database and enables natural language querying using Groq's Llama3 model.

## Features

- 📤 **File Upload**: Support for CSV, JSON, and TXT email files
- 🗄️ **Vector Database**: Automatic conversion to ChromaDB vector database
- 🔍 **Natural Language Queries**: Query emails using plain English
- 🤖 **AI-Powered Responses**: Groq Llama3 integration for intelligent responses
- 📊 **Analytics Dashboard**: Built-in analytics and insights
- 🎨 **Streamlit Frontend**: User-friendly web interface

## Requirements

- Python 3.8+
- Groq API Key
- Minimum 1000 emails for optimal performance

## Installation

1. **Clone the repository**:
   ```bash
   cd email_analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### 1. Upload Email Dataset

- Navigate to the "Upload Emails" tab
- Upload your email file (CSV, JSON, or TXT format)
- The system will automatically:
  - Process and validate emails
  - Create vector embeddings
  - Store in local ChromaDB database

### 2. Query Emails

- Go to the "Query Emails" tab
- Enter natural language queries such as:
  - "Give me John Smith's email address"
  - "Find all emails from gmail.com"
  - "Show me emails related to marketing"

### 3. View Analytics

- Check the "Analytics" tab for dataset insights
- Run predefined analytics queries
- Get AI-powered analysis of your email data

## File Formats

### CSV Format
```csv
name,email,company,role
John Smith,john@example.com,Acme Corp,Manager
Jane Doe,jane@company.com,Tech Inc,Developer
```

### JSON Format
```json
[
  {
    "name": "John Smith",
    "email": "john@example.com",
    "company": "Acme Corp",
    "role": "Manager"
  }
]
```

### TXT Format
```
john@example.com
jane@company.com
admin@website.org
```

## Configuration

Edit `config.py` to customize:

- **GROQ_MODEL**: Change the Groq model (default: llama3-8b-8192)
- **EMBEDDING_MODEL**: Change the embedding model (default: all-MiniLM-L6-v2)
- **VECTOR_DB_PATH**: Change vector database storage location
- **MAX_FILE_SIZE**: Adjust maximum upload file size

## API Keys

### Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for an account
3. Navigate to API Keys section
4. Create a new API key
5. Add it to your `.env` file

## Architecture

```
Email File → Email Processor → Vector Database → LLM Handler → Streamlit UI
     ↓              ↓               ↓              ↓            ↓
   Parse &      Validate &      ChromaDB      Groq Llama3   User Interface
   Extract      Transform       Storage       Processing    & Interaction
```

## Components

- **EmailProcessor**: Handles file parsing and email validation
- **VectorDBManager**: Manages ChromaDB operations and vector storage
- **LLMHandler**: Interfaces with Groq API for AI responses
- **Streamlit App**: Provides the web interface

## Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not found"**
   - Ensure you've created a `.env` file with your API key
   - Check that the key is valid and active

2. **"No valid emails found"**
   - Verify your file format matches the expected structure
   - Check that email addresses are properly formatted

3. **Vector database errors**
   - Clear the database using the sidebar button
   - Ensure sufficient disk space for the vector database

4. **Slow performance**
   - Reduce the number of search results
   - Consider using a smaller dataset for testing

### Performance Tips

- For large datasets (10k+ emails), consider processing in batches
- Use specific queries for better search results
- Clear the database periodically to maintain performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration options
3. Open an issue on the repository