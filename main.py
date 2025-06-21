import openai
import yaml
import json
import os
from dotenv import load_dotenv
os.environ.pop("SSL_CERT_FILE", None)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
OPEN_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")


client = openai.OpenAI(api_key=OPEN_API_KEY)
def read_task_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def yaml_to_json(task_yaml):
    """Convert YAML data to JSON format for easier processing"""
    return json.dumps(task_yaml, indent=2)

def build_code_generation_prompt(task_yaml, parent_folder='.'):
    """Step 2: Create comprehensive prompt for code generation using task configuration"""
    # Get specific technical details
    api_url = task_yaml.get('model_information', {}).get('api_url', 'API_URL_NOT_SPECIFIED')
    dataset_path = task_yaml.get('dataset_description', {}).get('data_path', './data')
    output_type = task_yaml.get('model_information', {}).get('output_format', {}).get('type', 'unknown')
    input_keys = list(task_yaml['model_information']['input_format']['structure'].keys())

    # Feed previous attempt for better performance
    previous_code = task_yaml.get('previous_attempt', {}).get('code',{})
    previous_feedback = task_yaml.get('previous_attempt', {}).get('feedback',{})

    first_key = input_keys[0] if input_keys else 'data'
    
    # Convert YAML to JSON for better LLM understanding
    task_json = yaml_to_json(task_yaml)
    
    prompt = f"""
IMPORTANT INSTRUCTIONS FOR CODE GENERATION:
- The path will be the absolute path to the folder containing task.yaml, the data folder, and any additional files or folders. Use this path as the root for all file and folder access. Do not assume a subfolder unless it is specified in task.yaml.
- The dataset may have a specific folder hierarchy (e.g., subfolders for classes, speakers, or other groupings). DO NOT simply process all files in the root folder. Instead, process files and folders according to the structure and requirements described in task.yaml (for example, only use files in certain subfolders, or follow the class/subclass structure as described).
- If the path does not exist, show an error message in the UI.

Generate a complete, production-ready Streamlit application based on these requirements:

TASK CONFIGURATION (JSON):
{task_json}

TECHNICAL SPECS:
- API Endpoint: {api_url}
- Dataset Path: {dataset_path}
- Output Format: {output_type}
- Parent Folder: {parent_folder}, include task.yaml, data (model input), and other addtional config file (.json, ...)
- Previous Attempt: {previous_code, previous_feedback}
- Do not join the env.env file

CRITICAL PATH REQUIREMENTS:
- All file paths should be relative to the parent folder: {parent_folder}
- Data files should be accessed from: {os.path.join(parent_folder, 'data') if dataset_path == './data' else dataset_path}
- Environment files should be loaded from: {os.path.join(parent_folder, 'env.env')}
- Configuration files should be read from the parent folder context

CRITICAL API RULES:
1. Payload structure: Use only {input_keys} as keys, not descriptions
   - Example: {{{first_key}: "actual_value"}}
2. Response handling: API response has "data" field containing {output_type} with a list wrapper
   - Get response: response_json = response.json()
   - There are 2 circumstances:
        - Example: 
        def inspect_structure(data):
            if isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    return data
                elif all(isinstance(item, list) and all(isinstance(sub, dict) for sub in item) for item in data):
                    return data[0]
        - First, there's no list wrapper, we just use data = response_json["data"]
        - Second, there's a list wrapper, we use data = response_json["data"][0] to unwrap the wrapper
        We have to try the first, then check if the type(data) = {output_type}, then we stop, else we check the 2nd circumstance
   - The data be in format: {output_type}

IMPLEMENTATION REQUIREMENTS:
- Complete Streamlit app with all imports
- Proper error handling and validation
- User-friendly interface with progress indicators
- The st.image() function should no longer use the deprecated use_column_width=True parameter. Instead, always use use_container_width=True for resizing images to fit the layout. Do not include deprecated or removed arguments
- Do not use the deprecated .render() method on Styler objects — it has been removed since pandas 1.3. Instead, always use .to_html() when exporting styled DataFrames to HTML. Ensure your code is compatible with modern pandas versions.
- Do not use the deprecated ImageDraw.textsize() method. Instead, use textbbox() if you need the bounding box (width and height) of the text, or textlength() if you only need the text length. Ensure all code is compatible with the latest Pillow versions.
- Carefuly check the hashable error.
- Caching with @st.cache_data where appropriate
- Bug-free, production-ready code only
- No explanations or markdown formatting

Generate the complete Python code now.
"""
    return prompt

def build_review_prompt(generated_code, task_yaml, parent_folder='.'):
    """Step 3: Create prompt for final code review and bug fixing"""
    api_url = task_yaml.get('model_information', {}).get('api_url', 'API_URL_NOT_SPECIFIED')
    output_type = task_yaml.get('model_information', {}).get('output_format', {}).get('type', 'unknown')
    input_keys = list(task_yaml['model_information']['input_format']['structure'].keys())
    first_key = input_keys[0] if input_keys else 'data'
    dataset_path = task_yaml.get('dataset_description', {}).get('data_path', './data')
    review_prompt = f"""
Review this generated Streamlit code and fix ALL bugs/issues:

GENERATED CODE:
```python
{generated_code}
```

CONTEXT: 
- API Endpoint: {api_url}
- Output Format: {output_type}
- Parent Folder: {parent_folder}
- Dataset Path: {dataset_path}

CRITICAL PATH CHECKS:
- Verify all file paths are correctly constructed relative to parent folder: {parent_folder}
- Data files should be accessed from: {os.path.join(parent_folder, 'data') if dataset_path == './data' else dataset_path}
- Environment files should be loaded from: {os.path.join(parent_folder, 'env.env')}
- Ensure proper path handling across different operating systems

CRITICAL CHECKS:
1. **API Integration**: 
   - Payload uses field names only (not descriptions)
   - Access pattern: 
        - If there's a wrapper : response_json = response.json() -> data = response_json["data"][0] -> results = data
        - If there's no wrapper: response_json = response.json() -> data = response_json["data"] -> results = data
        - Make sure the results is {output_type} format
   - Proper error handling for API calls

2. **Code Quality & Syntax**:
   - CRITICAL: Check for ALL syntax errors (indentation, brackets, quotes, colons, etc.)
   - Verify ALL imports are present and correct (io, sys, os, json, pandas, streamlit, requests, etc.)
   - Verify the validity of all functions from external libraries - some may be deprecated or removed
   - Check for undefined variables, functions, or methods
   - Ensure proper variable naming and scope
   - Validate all function calls and method invocations
   - Proper Streamlit components and caching
   - User-friendly error messages and input validation
   - Fix any Python syntax issues that would prevent execution

3. **Streamlit Best Practices**:
   - Fix any deprecated Streamlit parameters or methods
   - Ensure all Streamlit components use current API

4. **Functionality**:
   - Complete workflow implementation
   - File handling and data processing
   - UI components work correctly

OUTPUT REQUIREMENTS:
- If perfect: Return exactly "CODE_APPROVED"
- If issues found: Return ONLY the complete corrected Python code
- NO markdown formatting (```, ```python, etc.)
- NO explanations, comments about fixes, or descriptions
- NO code block markers or language indicators
- The output must be clean, executable Python code that can be directly saved to a .py file
- Ensure the code is syntactically correct and will run without Python errors
"""
    return review_prompt

def main(parent_folder='.'):
    print("🚀 Starting Multi-Stage Code Generation Pipeline...")
    
    # Construct task YAML path from parent folder
    task_yaml_path = os.path.join(parent_folder, 'task.yaml')
    
    # Load task configuration
    task_yaml = read_task_yaml(task_yaml_path)
    output_type = task_yaml['model_information']['output_format']['type']
    print("📖 Task YAML loaded successfully")
    
    # Stage 1: Generate code using task configuration
    print("\n🔨 Stage 1: Generating Streamlit application code...")
    code_prompt = build_code_generation_prompt(task_yaml, parent_folder)
    generated_code = call_llm(code_prompt)
    
    if generated_code is None:
        print("❌ Failed to generate code")
        return None
    
    # Stage 2: Review and fix the generated code
    print("\n🔍 Stage 2: Reviewing and fixing generated code...")
    review_prompt = build_review_prompt(generated_code, task_yaml, parent_folder)
    reviewed_code = call_llm(review_prompt)
    
    if reviewed_code is None:
        print("❌ Failed to review code, using initial version")
        final_code = generated_code
    else:
        # Check if code needs fixing or is approved
        if reviewed_code.strip() == "CODE_APPROVED":
            print("✅ Generated code approved without changes")
            final_code = generated_code
        else:
            print("🔧 Code issues found and fixed")
            final_code = reviewed_code    
    # Clean markdown markers if present
    final_code = clean_generated_code_str(final_code)
    print("\n🎉 Pipeline completed successfully!")
    return final_code

def call_llm(prompt):
    """Call OpenAI API with the given prompt"""
    try:
        response = client.chat.completions.create(
            model=OPEN_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")
        return None

def clean_generated_code_str(code_str):
    """Remove first and last lines from generated code string if they are markdown markers."""
    lines = code_str.splitlines(True)
    if len(lines) >= 2:
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
    return ''.join(lines)

if __name__ == "__main__":
    # For standalone CLI use
    code = main()
    with open('generated_ui.py', 'w', encoding='utf-8') as f:
        f.write(code)
