import os
from mistralai import Mistral


class RAGEngine:
    """
    Handles the full RAG pipeline:
    retrieval + prompt construction + LLM generation.
    """

    def __init__(self, vector_store):
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.vector_store = vector_store
        self.model = "mistral-small-latest"
        self.conversation_history = []

    def build_prompt(self, query, retrieved_chunks):
        """Build a prompt with retrieved context."""
        context = "\n\n".join([
            f"[Page {c['chunk']['page']}]: {c['chunk']['text']}"
            for c in retrieved_chunks
        ])

        system_prompt = """You are a helpful assistant that answers questions 
based strictly on the provided document context. 

Rules:
- Answer only based on the provided context
- If the answer is not in the context, say "I could not find this information in the document"
- Always cite the page number(s) where you found the answer
- Be concise and precise
- Format your response clearly"""

        user_message = f"""Context from the document:
{context}

Question: {query}

Answer based on the context above, citing page numbers:"""

        return system_prompt, user_message

    def answer(self, query, top_k=4):
        """Generate an answer using RAG."""
        # Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(query, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": "No relevant content found in the document.",
                "sources": [],
                "retrieved_chunks": []
            }

        # Build prompt
        system_prompt, user_message = self.build_prompt(query, retrieved_chunks)

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Generate response
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                *self.conversation_history
            ]
        )

        answer_text = response.choices[0].message.content

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": answer_text
        })

        # Extract unique source pages
        sources = sorted(set([
            c["chunk"]["page"] for c in retrieved_chunks
        ]))

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks
        }

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []