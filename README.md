# Loggworks Chatbot

This project implements a Loggworks chatbot that leverages AI to parse natural language commands and manage job-related tasks. The core of this application is built around an agent-based architecture, allowing for intelligent interpretation and execution of user requests.

## Key Features:

- **Natural Language Understanding**: The chatbot can understand and process user commands related to job scheduling, assignment, cancellation, and listing.
- **Structured Command Extraction**: Utilizes `langchain` and `pydantic` to extract structured `LoggworksCommand` objects from free-form text inputs.
- **Agent-Based Processing**: The `parse_job_prompts_with_ai` function acts as an intelligent agent, using an LLM (e.g., `gpt-4o-mini`) to interpret user requests and convert them into actionable commands.
- **Contextual Memory**: Employs `ConversationBufferMemory` to maintain conversation history, enabling the agent to resolve ambiguous references (e.g., "cancel it") based on previous interactions.
- **Multi-Agent Potential**: While currently a single-agent system, the structured command output and modular design lay the groundwork for future multi-agent interactions. For instance, a scheduling agent could interact with a technician assignment agent to optimize job allocation.
- **Job Management**: Integrates with a CSV file (`Loggworks_Job_Log_Sample_100.csv`) to store and retrieve job details, allowing for dynamic updates and queries.
- **Gradio Interface**: Provides an intuitive web-based chat interface using Gradio for easy interaction.

## How it Works:

1. **User Input**: Users interact with the chatbot through a Gradio interface, providing natural language commands or questions.
2. **AI-Powered Parsing**: The `loggworks_chatbot` function receives the user input and passes it to `parse_job_prompts_with_ai`. This function, acting as an agent, uses a `ChatOpenAI` model with a structured output parser to identify and extract `LoggworksCommand` objects.
3. **Contextual Awareness**: The agent utilizes `ConversationBufferMemory` to consider past turns in the conversation, improving its ability to understand and respond to follow-up questions or commands.
4. **Command Execution**: Based on the extracted commands, the system performs actions such as:
    - Retrieving job information using `get_jobs_by_multiple_inputs`.
    - Generating and logging new job entries or updates to `loggworks_commands.json` via `generate_command_json`.
5. **Conversational Response**: Finally, `generate_response` crafts a friendly and informative reply to the user, incorporating job details and suggesting next steps.

## Setup and Installation:

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd Loggworks
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**:
   Create a `.env` file in the root directory and add your OpenAI API key and model name:
   ```
   OPENAI_API_KEY="your_openai_api_key"
   OPENAI_MODEL_NAME="gpt-4o-mini" # or other compatible model
   ```
5. **Run the application**:
   ```bash
   python app.py
   ```
   The Gradio interface will be accessible in your browser, usually at `http://127.0.0.1:7860`.

## Docker Setup:

To run the application using Docker:

1. **Build the Docker image**:
   ```bash
   docker build -t loggworks-chatbot .
   ```
2. **Run the Docker container**:
   ```bash
   docker run -p 7860:7860 --env-file .env loggworks-chatbot
   ```
   The chatbot will be available at `http://localhost:7860`.

## Future Enhancements:

- **Multi-Agent Orchestration**: Implement a more sophisticated multi-agent system where different agents specialize in tasks like scheduling, resource allocation, and customer communication.
- **Database Integration**: Replace CSV with a more robust database solution for persistent storage and complex queries.
- **Advanced AI Models**: Integrate with more advanced LLMs for improved natural language understanding and response generation.