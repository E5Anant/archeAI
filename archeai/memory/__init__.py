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
        assistant_name:str = "Agent",
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
        self.name = assistant_name

        self.chat_buffer = []
        self.response_count = 0
        self.summarization_prompt = """
                You are an AI assistant responsible for creating a concise and coherent summary of the following conversation. 
                Please retain key points, topics discussed, and important details such as names, user interests and instructions or feedbacks, while omitting unnecessary or repetitive information. 
                The summary should be easy to understand and represent the essence of the conversation.

                Conversation:
                {conversation}
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


    def _should_meta_summarize(self) -> bool:
        """
        Checks if the memory file has grown beyond a defined size limit or contains
        too many individual summaries, triggering meta-summarization.
        """
        try:
            file_size = os.path.getsize(self.memory_filepath)  # Get size of memory.txt in bytes
            
            # Define the size threshold (e.g., 100 KB) or a summary count limit (e.g., 50 summaries)
            size_threshold = 60 * 1024  # 60 KB
            
            if file_size > size_threshold:
                logging.debug(f"Memory file exceeded threshold. Size: {file_size}")
                return True
            return False
        except FileNotFoundError:
            logging.debug("Memory file not found!")
            return False

    def _meta_summarize(self) -> None:
        """
        Summarizes all the summaries in memory.txt and overwrites it with a single meta-summary.
        """
        try:
            with open(self.memory_filepath, "r", encoding="utf-8") as f:
                summaries = f.read().strip()  # Load all summaries
                
            if summaries:
                # Generate a meta-summary from all past summaries using LLM
                meta_summary = self._summarize_memory(summaries)
                if meta_summary:
                    # Overwrite memory.txt with the new meta-summary
                    with open(self.memory_filepath, "w", encoding="utf-8") as f:
                        f.write(meta_summary)
                    self.memory_entries.clear()
                    self.memory_entries = [meta_summary]
                    logging.debug("Memory file meta-summarized successfully.")
                else:
                    logging.warning("Meta-summary failed, memory file not updated.")
            else:
                logging.debug("No summaries available for meta-summarization.")
        except IOError as e:
            logging.error(f"Error during meta-summarization: {e}")

    def _summarize_memory(self, text:str) -> str:
        """
        Summarizes a given text input. Uses LLM to generate summaries.
        """
        prompt = f"""
            Provide a brief summary of the following context. Focus on essential details, facts, and key points.
            Your output should be short and concise, representing the most important aspects of the context .
            And Consisting of user interests, names, topics, and instructions or feedbacks.

            Context:
            {text}
            """
        # Perform LLM summarization using the provided system prompt
        return self.llm.run(prompt)

    def _check_memory_size(self) -> None:
        """
        Checks the size of memory.txt and triggers meta-summarization if necessary.
        """
        if self._should_meta_summarize():
            logging.debug("Triggering meta-summarization due to memory size.")
            self._meta_summarize()
        else:
            logging.debug("Memory size within limits, no summarization needed.")

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
                # Load each line, stripping empty lines and ignoring corrupt or invalid entries
                self.memory_entries = [line.strip() for line in f if line.strip()]
            
            logging.debug(f"Loaded {len(self.memory_entries)} valid memory entries.")
            if self.memory_entries:
                self._update_tfidf_matrix()  # Update the TF-IDF matrix based on loaded entries
            else:
                logging.debug("No valid memory entries found.")
                self._update_tfidf_matrix()  # Ensure an empty matrix is initialized
        except FileNotFoundError:
            logging.warning(f"Memory file not found: {self.memory_filepath}. Starting with empty memory.")
            self.memory_entries = []
            self._update_tfidf_matrix()  # Initialize an empty TF-IDF matrix
        except Exception as e:
            logging.error(f"Error loading memory: {e}")
            self.memory_entries = []  # Clear any partial or corrupt data
            self._update_tfidf_matrix()  # Ensure the matrix is re-initialized

    def _update_tfidf_matrix(self):
        """Updates the TF-IDF matrix based on the current memory entries."""
        if self.memory_entries and any(entry.strip() for entry in self.memory_entries):
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.memory_entries)
                logging.debug("TF-IDF matrix updated successfully.")
            except ValueError as e:
                logging.error(f"TF-IDF update failed due to invalid entries: {e}")
                self.tfidf_matrix = None
        else:
            logging.debug("No valid memory entries available for TF-IDF update.")
            self.tfidf_matrix = None

    def _retrieve_memories(self, query: str, top_k: int = 3) -> list:
        """Memory retrieval with improved handling of similarity scores."""
        if self.tfidf_matrix is not None and self.tfidf_matrix.shape[0] > 0:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix)
            top_indices = similarity_scores.argsort()[0][-top_k:][::-1]
            
            # Filter out entries with very low similarity scores
            return [
                self.memory_entries[i]
                for i in top_indices
                if similarity_scores[0, i] > 0.1
            ]
        
        logging.debug("TF-IDF matrix is empty or invalid. No memories to retrieve.")
        return []

    def _should_summarize(self) -> bool:
        """
        Checks if the chat buffer has exceeded max_responses, 
        ensuring summarization doesn't occur too early.
        """
        # Check if buffer length exceeds the maximum response limit
        if self.response_count >= self.max_responses:
            logging.debug(f"Chat buffer reached max_responses: {len(self.chat_buffer)}")
            return True
        logging.debug(f"Chat buffer has {len(self.chat_buffer)} responses, below the threshold.")
        return False

    def _load_chat_history(self) -> None:
        """Loads the existing chat history and manages the chat buffer effectively."""
        try:
            with open(self.chat_filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            if len(lines) > 1:
                self.chat_buffer = lines[0:]  # Load chat history, skip system prompt
                # Summarize and clean up if chat buffer exceeds limits
                if len(self.chat_buffer) >= self.max_responses:
                    summary = self._summarize_chat()
                    self._save_memory(summary)
                    self.chat_buffer = []  # Clear buffer
                    self.response_count = 0  # Reset response count
                    recent_chat = "".join(lines[-self.max_responses:])  # Retain last N responses
                    self._write_to_chat_file(self.system_prompt + "\n" + "\n", mode="w")
            else:
                logging.debug("Chat file exists but is empty or contains only the system prompt.")
        except FileNotFoundError:
            logging.debug(f"Chat history file not found: {self.chat_filepath}. Starting fresh.")
        except IOError as e:
            logging.error(f"Error loading chat history: {e}")

    def _summarize_chat(self) -> str:
        """Summarizes the chat buffer."""
        if not self.chat_buffer:
            logging.debug("Chat buffer is empty, nothing to summarize.")
            return ""

        full_chat = "".join(self.chat_buffer[-self.max_responses:])
        length = min(250, max(50, len(full_chat) // 10))
        prompt = self.summarization_prompt.format(length=length, conversation=full_chat)

        try:
            # Use the llm to generate the summary
            summary = self.llm.run(prompt).strip()
            if len(summary) < 20:  # Ensure meaningful summary
                logging.warning("Generated summary is too short.")
            logging.debug(f"Chat summarized successfully: {summary}")
            return summary
        except Exception as e:
            logging.error(f"Error during chat summarization: {e}")
            return "Error: Unable to summarize the conversation."

    def _save_memory(self, summary: str) -> None:
        """Saves the summarized chat to the memory file and updates the TF-IDF matrix."""
        if summary.strip() and len(summary) > 20:  # Ensure valid and meaningful summary
            try:
                with threading.Lock():  # Ensure thread safety
                    with open(self.memory_filepath, "a", encoding="utf-8") as f:
                        f.write(summary + "\n")
                    self.memory_entries.append(summary)
                    self._check_memory_size()
                    self._update_tfidf_matrix()  # Update TF-IDF matrix
                    logging.debug("Summary saved to memory successfully.")
            except IOError as e:
                logging.error(f"Error saving memory: {e}")
        else:
            logging.debug("Skipping memory saving due to empty or invalid summary.")

    def update_chat_history(self, role: str, content: str, force: bool = False) -> None:
        """
        Updates the chat history with new messages and triggers summarization 
        when the response count exceeds the maximum.
        """
        if not self.status and not force:
            return

        new_history = f"{role}: {content}\n"
        self.chat_buffer.append(new_history)

        if role.startswith(self.name):
            self.response_count += 1
            logging.debug(f"Response count updated: {self.response_count}")

        # Check if summarization is required
        if self._should_summarize():
            summary = self._summarize_chat()  # Generate summary
            if summary:
                self._save_memory(summary)  # Save summary to memory
                self.chat_buffer = []  # Clear the buffer after summarizing
                self.response_count = 0  # Reset response count

                # Retain only recent chat history after summarizing
                recent_chat = "".join(self.chat_buffer[-self.max_responses:])
                self._write_to_chat_file(self.system_prompt + "\n" + "\n", mode="w")
                logging.debug(f"Chat summarized and reset. Recent chat preserved.")
            else:
                logging.warning("Summarization failed. Keeping chat buffer intact.")
        else:
            self._write_to_chat_file(new_history)
            logging.debug(f"Chat updated with new entry: {new_history.strip()}")
            
        # Always check if memory size exceeds threshold after any update
        self._check_memory_size()
 

    def gen_complete_prompt(self, current_turn: str) -> str:
        """Generates the complete prompt for the LLM with relevant memories and recent chat history."""
        relevant_memories = self._retrieve_memories(current_turn)
        memories_str = "\n".join([f"- {memory}" for memory in relevant_memories])

        recent_chat = "".join(self.chat_buffer[-self.max_responses:])  # Include the most recent chat buffer

        prompt = f"""Relevant Memories:
        {memories_str}

        Recent Conversation:
        {recent_chat}

        Current Turn:
        {current_turn}
        """

        return prompt