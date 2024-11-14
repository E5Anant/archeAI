import os
import threading
from threading import Thread
import sqlite3
from datetime import datetime
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer

class Memory:
    """
    Manages conversation history, memory summarization, and provides relevant
    context to the agent. Uses SQLite for storage and retrieval.

    This version includes:
    - Stores timestamps directly in the database for each message.
    - Retrieves chat history as a list of dictionaries.
    - Multiple memory retrieval strategies (recency, keyword-based).
    - Concurrent access handling with threading.
    - Meta-summarization to combine older memories when a maximum limit is reached.
    - TF-IDF for improved keyword-based memory retrieval.
    - Asynchronous TF-IDF updates for better responsiveness.
    - Deadlock prevention with consistent locking order.
    - Thread-safe SQLite connections to prevent errors. 
    """

    def __init__(
        self,
        llm,
        status: bool = True,
        max_responses: int = 30,
        max_memories: int = 10,
        history_dir: str = "memories",
        db_filename: str = "agent_memory.db",
        assistant_name: str = "Agent",
        system_prompt: str = "You are a helpful AI assistant.",
    ):
        self.llm = llm
        self.status = status
        self.max_responses = max_responses
        self.max_memories = max_memories
        self.db_filepath = os.path.join(history_dir, db_filename)
        self.system_prompt = system_prompt
        self.name = assistant_name

        self.chat_buffer = []
        self.response_count = 0
        self.conversation_start_time = None
        self.summarization_prompt = """
        You are an AI assistant tasked with summarizing a conversation. 
        Your summary should be divided into three sections:

        **Summary:** [A very brief summary of the conversation in no more than 2 sentences.]
        **Key Points:**
        - [Key point 1]
        - [Key point 2]
        - [Key point 3] 
        ...

        Focus on essential details, facts, and key points. Key points should include:
        * User instructions or requests.
        * Important information exchanged like names, topics, locations, etc. 
        * Decisions made or outcomes.
        * User feedback or sentiment.

        Conversation:
        {conversation}
        """
        self.meta_summarization_prompt = """
        You are an AI assistant tasked with summarizing multiple past conversations.
        These conversations are already summarized, but I need you to condense them 
        into one comprehensive summary.

        **Summary:** [A very brief summary of all the conversations combined.]
        **Key Points:**
        - [Key point 1]
        - [Key point 2]
        - [Key point 3] 
        ...

        Focus on the most important details and recurring themes across all conversations.

        Here are the conversation summaries:
        {conversation_summaries}
        """

        os.makedirs(history_dir, exist_ok=True)
        self.db_lock = threading.Lock()
        self._create_tables()

        # TF-IDF setup
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.memory_texts = []
        self.tfidf_update_lock = threading.Lock()  
        self._build_tfidf_index()

    def _create_tables(self):
        """Creates the necessary tables in the SQLite database."""
        with sqlite3.connect(self.db_filepath) as conn: 
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _build_tfidf_index(self):
        with self.db_lock:
            with sqlite3.connect(self.db_filepath) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT summary FROM memories")
                self.memory_texts = [row[0] for row in cursor.fetchall()]
        if self.memory_texts:
            self._update_tfidf_matrix()

    def _update_tfidf_matrix(self):
        """Updates the TF-IDF matrix. This function is designed to be thread-safe."""
        with self.tfidf_update_lock:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.memory_texts)

    def _update_tfidf_index_async(self, new_memory: str):
        """Updates the TF-IDF index asynchronously in a separate thread."""
        self.memory_texts.append(new_memory)
        self._update_tfidf_matrix()  

    def _get_memories_by_tfidf(self, query: str, limit: int = 3) -> list:
        if self.tfidf_matrix is None:
            return []
        with self.tfidf_update_lock:  
            query_vector = self.tfidf_vectorizer.transform([query])
            similarity_scores = (self.tfidf_matrix * query_vector.T).toarray()
            top_indices = similarity_scores.argsort(axis=0)[-limit:].flatten()[::-1]
        top_memories = [self.memory_texts[i] for i in top_indices]
        return top_memories

    def _add_memory(self, summary: str):
        with self.tfidf_update_lock:
            with self.db_lock:
                with sqlite3.connect(self.db_filepath) as conn:
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO memories (summary) VALUES (?)", (summary,))
                    conn.commit()
                    
                    # Do NOT call _manage_memory_limit here
            # Update TF-IDF index asynchronously after releasing db_lock 
            Thread(target=self._update_tfidf_index_async, args=(summary,)).start()

            # Call _manage_memory_limit AFTER releasing db_lock
            with sqlite3.connect(self.db_filepath) as conn:
                self._manage_memory_limit(conn)

    def _manage_memory_limit(self, conn):
        """Manages the memory limit. 
        
        Note: This method now takes a database connection as an argument.
        """
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        memory_count = cursor.fetchone()[0]
        if memory_count > self.max_memories:
            Thread(target=self._perform_meta_summarization).start()  

    def _perform_meta_summarization(self):
        # Create a new connection for this thread
        with sqlite3.connect(self.db_filepath) as conn: 
            with self.tfidf_update_lock:
                with self.db_lock:
                    cursor = conn.cursor() 
                    cursor.execute("SELECT timestamp, summary FROM memories ORDER BY timestamp ASC")
                    memories_to_summarize = cursor.fetchall()

                    conversation_summaries = "\n\n".join(
                        [memory[1] for memory in memories_to_summarize]
                    )
                    prompt = self.meta_summarization_prompt.format(
                        conversation_summaries=conversation_summaries
                    )
                    try:
                        meta_summary = self.llm.run(prompt).strip()

                        # Delete old memories
                        cursor.execute("DELETE FROM memories")
                        
                        # Insert the meta-summary 
                        cursor.execute(
                            "INSERT INTO memories (timestamp, summary) VALUES (?, ?)",
                            (datetime.now(), meta_summary),
                        )
                        conn.commit()

                        # Update TF-IDF index
                        self.memory_texts = [meta_summary]
                        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.memory_texts)
                    except Exception as e:
                        print(f"Error during meta-summarization: {e}")

    def _get_recent_memories(self, limit: int = 5) -> list:
        with self.db_lock:
            with sqlite3.connect(self.db_filepath) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT summary FROM memories ORDER BY timestamp DESC LIMIT ?", (limit,)
                )
                return [row[0] for row in cursor.fetchall()]

    def _add_chat_message(self, role, content):
        with self.db_lock:
            with sqlite3.connect(self.db_filepath) as conn:
                cursor = conn.cursor()
                # Convert timestamp to a string format (e.g., ISO format)
                timestamp = datetime.now().isoformat()
                cursor.execute(
                    "INSERT INTO chat_history (timestamp, role, content) VALUES (?, ?, ?)",
                    (timestamp, role, content),
                )
                conn.commit()

    def _get_recent_chat_messages(self, limit: int = 10) -> list:
        with self.db_lock:
            with sqlite3.connect(self.db_filepath) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT timestamp, role, content FROM chat_history ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
                messages = [
                    {
                        "timestamp": datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f"),
                        "role": role,
                        "content": content,
                    }
                    for timestamp, role, content in cursor.fetchall()
                ]
                return messages

    def _summarize_chat(self) -> Optional[str]:
        if not self.chat_buffer:
            return None

        conversation = "\n".join(
            [
                f"{message['role']}: {message['content']} [{message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}]"
                for message in self.chat_buffer
            ]
        )
        prompt = self.summarization_prompt.format(conversation=conversation)
        try:
            summary = self.llm.run(prompt).strip()
            return summary
        except Exception as e:
            print(f"Error during chat summarization: {e}")
            return None

    def _clear_chat_buffer(self):
        self.chat_buffer.clear()
        self.response_count = 0
        self.clear_chat_history()

    def clear_chat_history(self):
        with self.db_lock:
            with sqlite3.connect(self.db_filepath) as conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM chat_history")
                    conn.commit()
                except Exception as e:
                    print(f"Error clearing chat history from database: {e}")

    def update_chat_history(self, role: str, content: str) -> None:
        if not self.status:
            return

        if role != self.name and not self.chat_buffer:
            self.conversation_start_time = datetime.now()

        self.chat_buffer.append(
            {"timestamp": datetime.now(), "role": role, "content": content}
        )
        self._add_chat_message(role, content)

        if role.startswith(self.name):
            self.response_count += 1

        if self.response_count >= self.max_responses:
            summary = self._summarize_chat()
            if summary:
                self._add_memory(summary)  # This will handle locking correctly
                self._clear_chat_buffer()

    def gen_complete_prompt(self, current_turn: str) -> str:
        recent_memories = self._get_recent_memories(limit=2)
        keyword_memories = self._get_memories_by_tfidf(current_turn, limit=3)
        memories_str = "\n".join(
            [f"- {memory}" for memory in recent_memories + keyword_memories]
        )

        recent_chat = "\n".join(
            [
                f"{message['role']}: {message['content']} [{message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}]"
                for message in self._get_recent_chat_messages(limit=self.max_responses)
            ]
        )

        prompt = f"""Relevant Memories:
        {memories_str}

        Recent Conversation:
        {recent_chat}

        Current Turn:
        {current_turn}
        """
        return prompt