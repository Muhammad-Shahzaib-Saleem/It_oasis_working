from groq import Groq
from typing import List, Dict, Any
from config import GROQ_API_KEY, GROQ_MODEL
from config import QWEN_MODEL,OPEN_ROUTER_API,DEEP_SEEK_MODEL,LLAMA_MODEL
from openai import OpenAI

class LLMHandler:
    """Handle LLM interactions using Groq API with Llama3"""
    
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # self.client = Groq(api_key=GROQ_API_KEY)
        # self.model = GROQ_MODEL

        self.client =OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPEN_ROUTER_API,
        )

        self.model = DEEP_SEEK_MODEL
    
    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate response based on query and search results"""
        try:
            # Prepare context from search results
            context = self._prepare_context(search_results)
            
            # Create prompt
            prompt = self._create_prompt(query, context)
            
            # Generate response using Groq
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent email analysis assistant. You help users find and analyze email data based on their queries. Provide accurate, helpful, and well-formatted responses based on the email data provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare context from search results"""
        if not search_results:
            return "No relevant email data found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):  # Limit to top 5 results
            metadata = result.get('metadata', {})
            distance = result.get('distance', 0)
            
            # Format email information
            email_info = []
            for key, value in metadata.items():
                if value and str(value).strip():
                    email_info.append(f"{key}: {value}")
            
            if email_info:
                context_parts.append(f"Email {i} (Relevance: {1-distance:.2f}):\n" + "\n".join(email_info))
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM"""
        prompt = f"""
Based on the following email data, please answer the user's query accurately and helpfully.

User Query: {query}

Email Data:
{context}

Instructions:
1. Answer the query based on the provided email data
2. If looking for specific emails, provide the email addresses and relevant details
3. If the query asks for analysis or patterns, provide insights based on the data
4. If no relevant data is found, clearly state that
5. Format your response clearly and professionally
6. Include specific email addresses when relevant to the query

Response:"""
        
        return prompt
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent of the user query"""
        query_lower = query.lower()
        
        intent_analysis = {
            'type': 'general',
            'keywords': [],
            'is_email_search': False,
            'is_person_search': False,
            'is_domain_search': False,
            'is_analysis_request': False
        }
        
        # Check for email search patterns
        if any(keyword in query_lower for keyword in ['email', 'address', '@']):
            intent_analysis['is_email_search'] = True
            intent_analysis['type'] = 'email_search'
        
        # Check for person search patterns
        if any(keyword in query_lower for keyword in ['person', 'name', 'who', 'someone']):
            intent_analysis['is_person_search'] = True
            intent_analysis['type'] = 'person_search'
        
        # Check for domain search patterns
        if any(keyword in query_lower for keyword in ['domain', '.com', '.org', '.net']):
            intent_analysis['is_domain_search'] = True
            intent_analysis['type'] = 'domain_search'
        
        # Check for analysis patterns
        if any(keyword in query_lower for keyword in ['analyze', 'analysis', 'pattern', 'trend', 'statistics', 'count', 'how many']):
            intent_analysis['is_analysis_request'] = True
            intent_analysis['type'] = 'analysis'
        
        return intent_analysis
    
    def generate_summary(self, emails_data: List[Dict[str, Any]]) -> str:
        """Generate a summary of the email dataset"""
        try:
            if not emails_data:
                return "No email data available for summary."
            
            # Prepare basic statistics
            total_emails = len(emails_data)
            
            # Extract domains
            domains = set()
            for email in emails_data:
                if 'email' in email:
                    domain = email['email'].split('@')[-1] if '@' in email['email'] else ''
                    if domain:
                        domains.add(domain)
            
            # Create summary prompt
            sample_emails = emails_data[:3]  # Sample for analysis
            sample_text = "\n".join([email.get('text_content', '')[:200] for email in sample_emails])
            
            prompt = f"""
Analyze this email dataset and provide a comprehensive summary:

Total Emails: {total_emails}
Unique Domains: {len(domains)}
Top Domains: {', '.join(list(domains)[:5])}

Sample Email Data:
{sample_text}

Please provide:
1. Dataset overview
2. Key patterns or insights
3. Data quality assessment
4. Potential use cases for this data

Keep the summary concise but informative.
"""
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst specializing in email dataset analysis. Provide clear, professional summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=512
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"