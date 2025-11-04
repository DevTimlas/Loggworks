import warnings
warnings.filterwarnings('ignore')
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic.v1 import BaseModel, Field
from datetime import datetime, date
# import logging
from loguru import logger
import json
from dotenv import load_dotenv
import gradio as gr

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define the structured output schema for Loggworks commands
class LoggworksCommand(BaseModel):
    command: str = Field(description="Loggworks command (e.g., 'schedule', 'reschedule', 'assign', 'reassign', 'cancel', 'complete', 'list', 'reactivate')")
    client_name: str = Field(description="Client name (e.g., 'Acme Corp'), empty if not specified")
    job_type: str = Field(description="Job type (e.g., 'Plumbing', 'Repair')")
    date: str = Field(description="Scheduled date in YYYY-MM-DD format (e.g., '2025-06-10')")
    time: str = Field(description="Scheduled time in HH:MM format (e.g., '14:00')")
    technician: str = Field(description="Assigned technician (e.g., 'Sarah Lee'), empty if not specified", default="")
    notes: str = Field(description="Additional notes (e.g., 'Ongoing maintenance'), default empty", default="")
    job_id: str = Field(description="Job ID (e.g., '4589'), default empty", default="")

class JobData(BaseModel):
    commands: List[LoggworksCommand] = Field(description="List of Loggworks commands")

def load_csv(file_path: str = "Loggworks_Job_Log_Sample_100.csv") -> pd.DataFrame:
    """Load the CSV file into a pandas DataFrame with expected columns."""
    expected_columns = ['Job_ID', 'Client_Name', 'Job_Type', 'Status', 'Assigned_Technician', 
                        'Scheduled_Date', 'Scheduled_Time', 'Completion_Percentage', 'Completed_Date', 'Notes']
    try:
        df = pd.read_csv(file_path)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ''
        if 'Scheduled_Date' in df.columns:
            df['Scheduled_Date'] = pd.to_datetime(df['Scheduled_Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        if 'Completed_Date' in df.columns:
            df['Completed_Date'] = pd.to_datetime(df['Completed_Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        return df.fillna('')
    except FileNotFoundError:
        logger.error(f"CSV file {file_path} not found")
        return pd.DataFrame(columns=expected_columns)
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return pd.DataFrame(columns=expected_columns)

def return_names():
    df = load_csv()
    return df[['Client_Name', 'Assigned_Technician']].to_dict('records')

def parse_job_prompts_with_ai(text_blocks: List[str], memory: ConversationBufferMemory) -> Dict[str, Any]:
    """Parse job-related text prompts into structured Loggworks commands using AI."""
    if not isinstance(text_blocks, list):
        logger.error("Input must be a list")
        return {"commands": []}

    if not text_blocks:
        return {"commands": []}

    full_text = "\n".join(text_blocks)
    llm = ChatOpenAI(model=os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini'), temperature=0, max_retries=3)
    memory_context = memory.load_memory_variables({})["history"]

    all_names = return_names()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert at extracting structured job commands for Loggworks. Parse the input into structured commands in JSON format. Each command includes:
        - "command": The action (e.g., 'schedule', 'reschedule', 'assign', 'reassign', 'cancel', 'complete', 'list', 'reactivate').
        - "client_name": The client name (e.g., 'John Smith'), set to empty string if not specified.
        - "job_type": The job type (e.g., 'Plumbing', 'Repair').
        - "date": The scheduled date in YYYY-MM-DD format (e.g., '2025-06-10').
        - "time": The scheduled time in HH:MM format (e.g., '14:00').
        - "technician": The assigned technician (e.g., 'Sarah Lee'), set to empty string if not specified.
        - "notes": Additional details (e.g., 'Ongoing maintenance'), default empty.
        - "job_id": The job ID (e.g., '4589'), default empty.
        
        Guidelines:
        - Convert relative dates (e.g., 'next Friday') to absolute dates assuming today is {today_date}.
        - Use conversation history to resolve ambiguous inputs (e.g., 'cancel it' refers to the last mentioned job ID).
        - If client_name or technician is not specified, set to empty string.
        - Handle diverse inputs (e.g., "schedule a plumbing job on 2025-06-10 at 14:00", "list all plumbing jobs", "hi").
        - Return empty commands list for conversational inputs (e.g., 'hi').
        - If multiple commands are present, extract all of them.
        - For 'list' commands, set only relevant fields based on input.
        - For notes-based searches, include notes keywords in the 'notes' field.
         
        FYI: Available clients and technicians: {all_names}

        Conversation History: {memory_context}
        User Input: {text}
        Today: {today_date}
        """),
        ("human", "{text}")
    ])

    chain = prompt | llm.with_structured_output(JobData)
    try:
        today = date.today()
        today_date = today.strftime("%Y-%m-%d")
        result = chain.invoke({"text": full_text, "today_date": today_date, "memory_context": memory_context, "all_names": all_names})
        logger.info(f"Parsed commands: {result.dict()}")
        return result.dict()
    except Exception as e:
        logger.error(f"LLM parsing failed: {str(e)}")
        return {"commands": []}

def search_jobs_by_notes(keywords: str, memory: ConversationBufferMemory) -> List[Dict[str, Any]]:
    """Search job rows by keywords in Notes using AI."""
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, max_retries=3)
    memory_context = memory.load_memory_variables({})["history"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Extract keywords from user input for searching job notes. Return a list of keywords in JSON format.
        Guidelines:
        - Extract specific terms (e.g., 'ongoing maintenance').
        - Ignore irrelevant words (e.g., 'list', 'all').
        - Use conversation history for context.
        - Return empty list if no keywords found.

        Conversation History: {memory_context}
        User Input: {text}
        """),
        ("human", "{text}")
    ])

    class Keywords(BaseModel):
        keywords: List[str] = Field(description="List of keywords or phrases to search in Notes")

    chain = prompt | llm.with_structured_output(Keywords)
    try:
        result = chain.invoke({"text": keywords, "memory_context": memory_context})
        keyword_list = result.keywords
        logger.info(f"Extracted keywords: {keyword_list}")
    except Exception as e:
        logger.error(f"Keyword extraction failed: {str(e)}")
        return []

    if not keyword_list:
        return []

    df = load_csv()
    if df.empty:
        return []

    try:
        query = df[df['Notes'].str.contains('|'.join(keyword_list), case=False, na=False)]
        if query.empty:
            logger.info(f"No jobs found with notes matching keywords: {keyword_list}")
            return []
        logger.info(f"Found {len(query)} jobs with notes matching keywords: {keyword_list}")
        return query.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error searching Notes: {str(e)}")
        return []

def get_jobs_by_multiple_inputs(
    client_names: Optional[Union[str, List[str]]] = None,
    job_types: Optional[Union[str, List[str]]] = None,
    statuses: Optional[Union[str, List[str]]] = None,
    scheduled_dates: Optional[Union[str, List[str]]] = None,
    technicians: Optional[Union[str, List[str]]] = None,
    job_ids: Optional[Union[int, str, List[Union[int, str]]]] = None,
    notes: Optional[str] = None,
    memory: Optional[ConversationBufferMemory] = None
) -> List[Dict[str, Any]]:
    """Retrieve rows from the CSV based on multiple optional inputs."""
    df = load_csv()
    if df.empty and not job_ids:
        return []

    query = df.copy()
    try:
        if client_names and client_names != [""]:
            if not isinstance(client_names, list):
                client_names = [client_names]
            query = query[query["Client_Name"].str.lower().isin([name.lower() for name in client_names if name])]

        if job_types and job_types != [""]:
            if not isinstance(job_types, list):
                job_types = [job_types]
            query = query[query["Job_Type"].str.lower().isin([jt.lower() for jt in job_types if jt])]

        if statuses and statuses != [""]:
            if not isinstance(statuses, list):
                statuses = [statuses]
            query = query[query["Status"].str.lower().isin([s.lower() for s in statuses if s])]

        if scheduled_dates and scheduled_dates != [""]:
            if not isinstance(scheduled_dates, list):
                scheduled_dates = [scheduled_dates]
            query = query[query["Scheduled_Date"].isin([d for d in scheduled_dates if d])]

        if technicians and technicians != [""]:
            if not isinstance(technicians, list):
                technicians = [technicians]
            query = query[query["Assigned_Technician"].str.lower().isin([t.lower() for t in technicians if t])]

        if job_ids and job_ids != [""]:
            if not isinstance(job_ids, list):
                job_ids = [job_ids]
            job_ids = [str(jid) for jid in job_ids if str(jid).isdigit()]
            query = query[query["Job_ID"].astype(str).isin(job_ids)]

        if notes and memory:
            notes_jobs = search_jobs_by_notes(notes, memory)
            if not notes_jobs:
                query = pd.DataFrame()
            else:
                notes_job_ids = [str(job['Job_ID']) for job in notes_jobs]
                query = query[query["Job_ID"].astype(str).isin(notes_job_ids)]

        if query.empty:
            input_summary = []
            if client_names and client_names != [""]: input_summary.append(f"client(s): {', '.join(client_names)}")
            if job_types and job_types != [""]: input_summary.append(f"job type(s): {', '.join(job_types)}")
            if statuses and statuses != [""]: input_summary.append(f"status(es): {', '.join(statuses)}")
            if scheduled_dates and scheduled_dates != [""]: input_summary.append(f"date(s): {', '.join(scheduled_dates)}")
            if technicians and technicians != [""]: input_summary.append(f"technician(s): {', '.join(technicians)}")
            if job_ids and job_ids != [""]: input_summary.append(f"job ID(s): {', '.join(map(str, job_ids))}")
            if notes: input_summary.append(f"notes: {notes}")
            logger.info(f"No jobs found for {', '.join(input_summary) if input_summary else 'no inputs provided'}")
            return []

        logger.info(f"Found {len(query)} jobs matching criteria")
        return query.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error querying CSV: {str(e)}")
        return []

def generate_command_json(commands: List[Dict[str, Any]], output_file: str = "loggworks_commands.json"):
    """Append Loggworks commands to a JSON file, skipping empty commands."""
    try:
        # Load existing JSON data if file exists
        existing_data = []
        if os.path.exists(output_file):
            with open(output_file, 'r') as jsonfile:
                existing_data = json.load(jsonfile)

        # Process new commands
        for cmd in commands:
            if not cmd.get('command') or cmd['command'] not in ['schedule', 'reschedule', 'assign', 'reassign', 'cancel', 'complete', 'reactivate']:
                logger.warning(f"Skipping invalid or empty command: {cmd}")
                continue
            job_id = cmd.get('job_id') or str(hash(f"{cmd['client_name']}{cmd['date']}{cmd['time']}{cmd['job_type']}") % 10000)
            status = 'Completed' if cmd['command'] == 'complete' else ('Active' if cmd['command'] in ['schedule', 'assign', 'reassign', 'reschedule', 'reactivate'] else 'Cancelled')
            completed_date = datetime.now().strftime('%Y-%m-%d') if cmd['command'] == 'complete' else ''
            completion_percentage = 100 if cmd['command'] == 'complete' else 0
            job_entry = {
                'Job_ID': job_id,
                'Client_Name': cmd.get('client_name', ''),
                'Job_Type': cmd.get('job_type', ''),
                'Status': status,
                'Assigned_Technician': cmd.get('technician', ''),
                'Scheduled_Date': cmd.get('date', ''),
                'Scheduled_Time': cmd.get('time', ''),
                'Completion_Percentage': completion_percentage,
                'Completed_Date': completed_date,
                'Notes': cmd.get('notes', '')
            }
            existing_data.append(job_entry)
            logger.info(f"Logged command {cmd['command']} for Job_ID {job_id} to {output_file}")

        # Write updated data to JSON
        with open(output_file, 'w') as jsonfile:
            json.dump(existing_data, jsonfile, indent=2)
    except Exception as e:
        logger.error(f"Error writing to JSON {output_file}: {str(e)}")

def generate_response(user_input: str, parsed_command: Dict[str, Any], job_data: List[Dict[str, Any]], memory: ConversationBufferMemory) -> str:
    """Generate a conversational response for user input."""
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, max_retries=3)
    memory_context = memory.load_memory_variables({})["history"]
    today = date.today()
    today_date = today.strftime("%Y-%m-%d")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a friendly Loggworks chatbot. Respond warmly, professionally, and concisely, using emojis (😊, ✔️, 📊). Handle job-related commands and casual conversation:
        - For commands (e.g., 'schedule', 'reschedule', 'assign', 'reassign', 'cancel', 'complete', 'list', 'reactivate'), confirm actions, include job details, and suggest follow-ups. For 'list' commands, summarize jobs concisely. If client_name or technician is not specified, indicate that the command applies broadly.
        - For casual inputs (e.g., 'hi'), respond conversationally, using history for context.
        - Use history for continuity (e.g., 'cancel it' refers to last job ID).
        - Handle empty commands as conversational.
        - Summarize jobs for weekly or notes-based queries.
        - Today is {today_date}.

        Conversation History: {memory_context}
        User Input: {user_input}
        Parsed Command: {parsed_command}
        Job Data: {job_data}
        """),
        ("human", "Generate a conversational response.")
    ])

    try:
        response = llm.invoke(prompt.format(
            today_date=today_date,
            user_input=user_input,
            parsed_command=parsed_command,
            job_data=job_data,
            memory_context=memory_context
        ))
        memory.save_context({"input": user_input}, {"output": response.content})
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "Oops, something went wrong! Let's try that again. 😅"

def loggworks_chatbot(message: str, history: List[Dict[str, str]], memory: ConversationBufferMemory = None) -> str:
    """Process user input for Gradio ChatInterface."""
    if memory is None:
        memory = ConversationBufferMemory(return_messages=True)

    if not message:
        return "Please enter a command or message! 😊"
    
    logger.info(f"User input: {message}")

    if message.lower() in ["exit", "quit"]:
        memory.clear()
        return "Goodbye! Chat cleared. Start a new conversation anytime! 👋"

    parsed_result = parse_job_prompts_with_ai([message], memory)
    responses = []
    job_data = []

    if parsed_result["commands"]:
        for command in parsed_result["commands"]:
            job_data = get_jobs_by_multiple_inputs(
                client_names=command["client_name"] if command["client_name"] else None,
                job_types=command["job_type"] if command["job_type"] else None,
                scheduled_dates=command["date"] if command["date"] else None,
                technicians=command["technician"] if command["technician"] else None,
                job_ids=command["job_id"] if command["job_id"] else None,
                notes=command["notes"] if command["notes"] else None,
                memory=memory
            )
            if command["command"] in ["schedule", "reschedule", "assign", "reassign", "cancel", "complete", "reactivate"]:
                generate_command_json([command])
            response = generate_response(message, {"commands": [command]}, job_data, memory)
            responses.append(response)
    else:
        job_data = get_jobs_by_multiple_inputs()
        response = generate_response(message, parsed_result, job_data, memory)
        responses.append(response)

    return ' '.join(responses)

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Launch Gradio ChatInterface
gr.ChatInterface(
    fn=loggworks_chatbot,
    type="messages",
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(
        placeholder="Ask about jobs or enter a command (e.g., 'schedule a plumbing job on 2025-06-10 at 14:00')",
        container=False,
        scale=7
    ),
    title="Loggworks Chatbot",
    description="Manage jobs with commands like 'schedule', 'reschedule', 'assign', 'reassign', 'cancel', 'complete', 'list', or 'reactivate'. Or just chat! 😊",
    theme="ocean",
    examples=["schedule a plumbing job on 2025-06-10 at 14:00", "list all jobs with ongoing maintenance", "cancel job id 4589"],
    cache_examples=False,
    css="footer {visibility: hidden}",
).launch(
    server_name="0.0.0.0",
    server_port=7860
)