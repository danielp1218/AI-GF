from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
import datetime
import time
import json
# dotenv is loaded in rime_agent.py, so no need to load again

INDEX_NAME = "ai-gf-memory"

class Memory:
    def __init__(self):
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Check if index exists, if not create it
        existing_indexes = [index.name for index in self.pinecone.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            self.pinecone.create_index(
                name=INDEX_NAME, 
                dimension=1536,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pinecone.Index(INDEX_NAME)
        self.agent = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def insert(self, text: list[str]):
        timestamp = str(datetime.datetime.now())
        embeddings = []
        for i in range(len(text)):
            embedding_response = self.agent.embeddings.create(input=(timestamp + ": " + text[i]), model="text-embedding-ada-002")
            embeddings.append({
                "id": timestamp + f"_{i}",
                "values": embedding_response.data[0].embedding,
                "metadata": {"text": text[i], "timestamp": timestamp}
            })
        self.index.upsert(embeddings)

    def query(self, text: str, top_k: int):
        query_response = self.agent.embeddings.create(input=text, model="text-embedding-ada-002")
        query_embedding = query_response.data[0].embedding
        return self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    def analyze_and_memorize(self, user_response: str, context: str = ""):
        analysis_prompt = f"""
        Analyze the following user response and extract valuable information that should be remembered for future conversations. 
        Focus on:
        - Personal preferences, interests, and opinions
        - Important facts about the user (background, relationships, goals)
        - Emotional context and sentiment
        - Any significant events or experiences mentioned
        - User's communication style and personality traits
        
        Context: {context}
        User Response: {user_response}
        
        Extract the most important pieces of information as a list of concise, memorable statements. 
        Each statement should be self-contained and useful for future reference.
        Format your response as a JSON array of strings, like this:
        ["statement 1", "statement 2", "statement 3"]
        
        If there's nothing particularly valuable to remember, return an empty array: []
        """
        
        try:
            response = self.agent.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing conversations and extracting valuable information to remember about users. Always respond with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3
            )
            
            analysis_result = response.choices[0].message.content.strip()
            
            # Handle markdown-formatted JSON
            if analysis_result.startswith("```json"):
                analysis_result = analysis_result.replace("```json", "").replace("```", "").strip()
            elif analysis_result.startswith("```"):
                analysis_result = analysis_result.replace("```", "").strip()
            
            memorable_items = json.loads(analysis_result)
            
            # save to memory
            if memorable_items and len(memorable_items) > 0:
                self.insert(memorable_items)
                return memorable_items
            else:
                return []
                
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print(f"Raw response: {analysis_result}")
            return []
        except Exception as e:
            print(f"Error analyzing user response: {e}")
            return []
    
    def update_from_conversation(self, conversation: str):
        return self.analyze_and_memorize(conversation, "Full conversation analysis")
    
    def get_relevant_memories(self, query: str, top_k: int = 5):
        results = self.query(query, top_k)
        
        if results and 'matches' in results:
            memories = []
            for match in results['matches']:
                if 'metadata' in match and 'text' in match['metadata']:
                    memories.append({
                        'text': match['metadata']['text'],
                        'score': match['score'],
                        'timestamp': match['metadata'].get('timestamp', 'unknown')
                    })
            return memories
        return []

