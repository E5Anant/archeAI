import os
import logging
from typing import Optional
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging (adjust level as needed)
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Memory:
    """
    Manages conversation history, memory summarization, and provides relevant 
    context to the agent using TF-IDF for semantic similarity search.
    """

    def __init__(
        self,
        llm, 
        status: bool = True,
        max_responses: int = 30,
        history_dir: str = "memories",
        memory_filepath: str = "agent_memory.txt",
        chat_filepath: str = "agent_chat.txt",
        update_file: bool = True,
        system_prompt: str = "You are a helpful AI assistant.",
    ):
        """Initializes the Memory object."""
        self.llm = llm
        self.status = status
        self.max_responses = max_responses
        self.history_dir = history_dir
        self.memory_filepath = os.path.join(history_dir, memory_filepath)
        self.chat_filepath = os.path.join(history_dir, chat_filepath)
        self.update_file = update_file
        self.system_prompt = system_prompt

        self.chat_buffer = []
        self.response_count = 0
        self.summarization_prompt = """You are a helpful AI assistant tasked with summarizing the following conversation in about {length} words.
        Conversation:
        {conversation}
        Summary:
        """

        # Ensure history directory and files exist 
        os.makedirs(self.history_dir, exist_ok=True)
        if not os.path.exists(self.memory_filepath):
            open(self.memory_filepath, 'w', encoding="utf-8").close() 
        if not os.path.exists(self.chat_filepath):
            with open(self.chat_filepath, 'w', encoding="utf-8") as f:
                f.write(self.system_prompt + "\n")  # Write system prompt initially

        # TF-IDF for memory
        self.memory_entries = []
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

        # Load existing data (memory and chat history)
        self._load_memory()
        self._load_chat_history()


    def _write_to_chat_file(self, content: str, mode: str = "a") -> None:
        """Thread-safe writing to the chat file with improved error handling."""
        if self.update_file:
            try:
                with threading.Lock():  # Ensuring thread safety
                    with open(self.chat_filepath, mode, encoding="utf-8") as fh:
                        fh.write(content)
            except IOError as e:
                logging.error(f"Error writing to chat file: {e}")

    def _load_memory(self) -> None:
        """Loads memory entries from the memory file and updates the TF-IDF matrix."""
        try:
            with open(self.memory_filepath, "r", encoding="utf-8") as f:
                self.memory_entries = [line.strip() for line in f if line.strip()]
            logging.debug(f"Loaded {len(self.memory_entries)} memory entries.")
            self._update_tfidf_matrix()  # Update the TF-IDF matrix after loading
        except FileNotFoundError:
            logging.debug(f"Memory file not found: {self.memory_filepath}. Starting fresh.")
            self.memory_entries = []
            self._update_tfidf_matrix()
        except IOError as e:
            logging.error(f"Error loading memory: {e}")

    def _update_tfidf_matrix(self):
        """Updates the TF-IDF matrix based on the current memory entries with improved checks."""
        if self.memory_entries and any(entry.strip() for entry in self.memory_entries):
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.memory_entries)
            except ValueError as e:
                logging.error(f"TF-IDF update failed due to invalid entries: {e}")
                self.tfidf_matrix = None
        else:
            logging.debug("No valid memory entries available for TF-IDF. Skipping update.")
            self.tfidf_matrix = None

    def _retrieve_memories(self, query: str, top_k: int = 3) -> list:
        """Improved memory retrieval with better handling of empty matrices."""
        if self.tfidf_matrix is not None and self.tfidf_matrix.shape[0] > 0:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix)
            top_indices = similarity_scores.argsort()[0][-top_k:][::-1]
            return [
                self.memory_entries[i] 
                for i in top_indices 
                if similarity_scores[0, i] > 0.1  # Exclude very low similarity entries
            ]
        logging.debug("TF-IDF matrix is empty or invalid. No memories to retrieve.")
        return []

    def _should_summarize(self) -> bool:
        """Checks if the chat buffer has reached the maximum allowed responses."""
        if len(self.chat_buffer) >= self.max_responses:
            logging.debug(f"Chat buffer exceeded max_responses: {len(self.chat_buffer)}")
            return True
        return False

    def _load_chat_history(self) -> None:
        """Loads the existing chat history, ensuring efficient buffer management."""
        try:
            with open(self.chat_filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    self.chat_buffer = lines[1:]  # Load chat history
                    if len(self.chat_buffer) >= self.max_responses:
                        summary = self._summarize_chat()
                        self._save_memory(summary)
                        self.chat_buffer = []
                        self.response_count = 0
                        self._write_to_chat_file(self.system_prompt, mode="w")
        except FileNotFoundError:
            logging.debug(f"Chat history file not found: {self.chat_filepath}. Starting fresh.")
        except IOError as e:
            logging.error(f"Error loading chat history: {e}")

    def _summarize_chat(self) -> str:
        """Summarizes the chat buffer and handles errors if summarization fails."""
        if not self.chat_buffer:
            logging.debug("Chat buffer is empty, nothing to summarize.")
            return ""

        full_chat = "".join(self.chat_buffer)
        length = min(250, max(50, len(full_chat) // 10))
        prompt = self.summarization_prompt.format(length=length, conversation=full_chat)

        try:
            self.llm.__init__(system_prompt="You are a helpful and concise AI assistant that summarizes text.")
            summary = self.llm.run(prompt).strip()
            if len(summary) < 20:  # Minimum summary length to ensure it's meaningful
                logging.warning("Generated summary is too short, retrying.")
                raise ValueError("LLM summary too short")
            self.llm.reset()  # Reset the LLM after running the summarization
            logging.debug(f"Chat summarized successfully: {summary}")
            return summary
        except Exception as e:
            logging.error(f"Error during chat summarization: {e}")
            return "Error: Unable to summarize the conversation."

    def _save_memory(self, summary: str) -> None:
        """Saves the summarized chat to the memory file and updates the TF-IDF matrix."""
        if summary.strip() and self.memory_filepath:
            try:
                with threading.Lock():  # Ensuring thread safety
                    with open(self.memory_filepath, "a", encoding="utf-8") as f:
                        f.write(summary + "\n")
                    self.memory_entries.append(summary)  # Append the new summary to the entries
                    self._update_tfidf_matrix()  # Update the TF-IDF matrix after adding new memory
                    logging.debug(f"Summary saved to memory: {summary}")
            except IOError as e:
                logging.error(f"Error saving memory: {e}")
        else:
            logging.debug("Skipping memory saving due to empty or invalid summary.")

    def update_chat_history(self, role: str, content: str, force: bool = False) -> None:
        """Updates the chat history and triggers summarization if the response count exceeds max_responses."""
        if not self.status and not force:
            return

        new_history = f"{role}: {content}\n"
        self.chat_buffer.append(new_history)

        if role.startswith("You:"):  # Increment response count only for user inputs
            self.response_count += 1
            logging.debug(f"Response count updated: {self.response_count}")
            
            # Check if we should summarize
            if self._should_summarize():
                summary = self._summarize_chat()  # Generate the summary
                self._save_memory(summary)  # Save to memory file
                self.chat_buffer = []  # Clear the buffer after summarizing
                self.response_count = 0  # Reset the response count
                logging.debug(f"Summary generated and chat buffer reset: {summary}")

                # Clear the chat file and write the system prompt and summary
                self._write_to_chat_file(self.system_prompt + "\n" + summary + "\n", mode="w")
            else:
                # Append to the chat file without summarization
                self._write_to_chat_file(new_history)
        else:
            # Append system or AI responses directly to the chat file
            self._write_to_chat_file(new_history) 

    def gen_complete_prompt(self, current_turn: str) -> str:
        """Generates the complete prompt for the LLM."""
        relevant_memories = self._retrieve_memories(current_turn)
        memories_str = "\n".join(
            [f"- {memory}" for memory in relevant_memories]
        )
        prompt = f"""Relevant Memories:
        {memories_str}

        Current Turn:
        {current_turn}
        """
        return prompt