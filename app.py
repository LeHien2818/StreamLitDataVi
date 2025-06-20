import streamlit as st
import os
import yaml
import tempfile
import main
import sys
import io
import shutil
import hashlib

# --- App Configuration ---
st.set_page_config(
    page_title="Dynamic UI Generator",
    page_icon="✨",
    layout="wide"
)

# --- Session State Initialization ---
# This dictionary will hold all the state for our internal router and data.
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        "view": "uploader",  # Can be 'uploader' or 'generated_app'
        "generated_code": None,
        "uploaded_file_name": None,
    }

# --- Helper Functions ---
def switch_view(view_name):
    """Function to switch the internal view."""
    st.session_state.app_state['view'] = view_name
    # Only set should_rerun if not already set, to avoid double rerun
    if not st.session_state.get('should_rerun', False):
        st.session_state['should_rerun'] = True

def cleanup_sample_folder():
    # Remove the extracted sample folder if it exists in session state
    sample_folder = st.session_state.app_state.get('sample_folder_path')
    if sample_folder and os.path.exists(sample_folder):
        try:
            shutil.rmtree(sample_folder)
        except Exception as e:
            print(f"Error deleting sample folder: {e}")
        st.session_state.app_state['sample_folder_path'] = None

def get_file_hash(uploaded_file):
    # Compute a hash of the uploaded file's content
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.sha256(file_bytes).hexdigest()

# --- Main App Logic ---

# Use the 'view' from state to decide what to render
current_view = st.session_state.app_state['view']

# ==============================================================================
# VIEW 1: The Uploader Interface
# ==============================================================================
if current_view == "uploader":
    # Clean up any previous sample folder if a new file is uploaded
    st.header("\U0001F4E4 Upload your Dataset (.zip)")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **Instructions:**
        - Upload a single .zip file containing everything needed for your task:
            - `task.yaml`
            - Dataset files/folders
            - Any additional files
        - The internal structure of the zip should match the relative paths referenced in `task.yaml`.
        """)
        task_bundle_zip = st.file_uploader(
            "Choose your task bundle as a .zip file (structure will be preserved)",
            type=['zip'],
            accept_multiple_files=False,
            help="Upload a single .zip file containing task.yaml, dataset, and any required files."
        )
        extract_dir = None
        extracted_files = []
        task_yaml_path = None
        task_yaml = None
        abs_task_yaml_path = None
        abs_paths_info = {}
        current_file_hash = None
        if task_bundle_zip is not None:
            import zipfile
            import tempfile
            import yaml as pyyaml
            import shutil
            import re
            # Extract zip to a temp directory
            extract_dir = tempfile.mkdtemp(prefix="task_bundle_")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                tmp_zip.write(task_bundle_zip.read())
                tmp_zip_path = tmp_zip.name
            with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                extracted_files = zip_ref.namelist()
            os.unlink(tmp_zip_path)
            # Find task.yaml in extracted files
            for f in extracted_files:
                if re.match(r'.*task\.ya?ml$', f):
                    task_yaml_path = os.path.join(extract_dir, f)
                    break
            if task_yaml_path and os.path.exists(task_yaml_path):
                with open(task_yaml_path, 'r', encoding='utf-8') as f:
                    task_yaml = pyyaml.safe_load(f)
                # Update all dataset paths in task_yaml to absolute paths inside extract_dir
                dataset_desc = task_yaml.get('dataset_description', {})
                updated = False
                for key in ['data_path', 'data_source']:
                    if key in dataset_desc:
                        rel_path = dataset_desc[key]
                        # Remove leading './' if present
                        rel_path_clean = rel_path[2:] if rel_path.startswith('./') else rel_path
                        abs_path = os.path.abspath(os.path.join(extract_dir, rel_path_clean))
                        dataset_desc[key] = abs_path
                        abs_paths_info[key] = abs_path
                        updated = True
                if updated:
                    # Write a new task.yaml with updated paths
                    abs_task_yaml_path = os.path.join(extract_dir, 'task_abs.yaml')
                    with open(abs_task_yaml_path, 'w', encoding='utf-8') as f:
                        pyyaml.dump(task_yaml, f, allow_unicode=True)
                    st.session_state.app_state['task_yaml_path'] = abs_task_yaml_path
                else:
                    # No dataset path found, use original
                    st.session_state.app_state['task_yaml_path'] = task_yaml_path
                st.session_state.app_state['task_yaml'] = task_yaml
                st.session_state.app_state['extract_dir'] = extract_dir
                st.session_state.app_state['extracted_files'] = extracted_files
                st.session_state.app_state['abs_paths_info'] = abs_paths_info
                st.session_state.app_state['data_path'] = task_yaml_path[:-10]
            else:
                # Reuse previous extraction
                task_yaml_path = st.session_state.app_state.get('task_yaml_path')
                task_yaml = st.session_state.app_state.get('task_yaml')
                abs_task_yaml_path = st.session_state.app_state.get('task_yaml_path')
                extracted_files = st.session_state.app_state.get('extracted_files', [])
                abs_paths_info = st.session_state.app_state.get('abs_paths_info', {})

    with col2:
        st.markdown("**Supported format:**")
        st.markdown("• .zip (must include task.yaml, dataset, and any required files)")
        st.markdown("**Bundle Contents**")
        if extract_dir:
            st.markdown(f"• Extracted to: `{extract_dir}`")
            if extracted_files:
                st.markdown("**Extracted Files:**")
                for f in extracted_files[:10]:
                    st.markdown(f"- {f}")
                if len(extracted_files) > 10:
                    st.markdown(f"...and {len(extracted_files)-10} more files.")
            if task_yaml_path:
                st.markdown(f"**Detected task.yaml:** `{task_yaml_path}`")
            if abs_paths_info:
                st.markdown(f"**Absolute dataset paths used:** `{st.session_state.app_state['data_path']}`")
        else:
            st.markdown("• No bundle zip uploaded yet")

    st.markdown("---")

    if 'task_yaml_path' in st.session_state.app_state and st.session_state.app_state['task_yaml_path']:
        st.subheader("\U0001F680 Ready to Generate UI Code")
        if st.button("\u2728 Generate UI", type="primary", use_container_width=True):
            with st.spinner("Generating UI Code... This may take a moment."):
                try:
                    # Use the extracted (and possibly updated) task.yaml for code generation
                    generated_code_string = main.main(st.session_state.app_state['data_path'])
                    # Store the generated code in session state
                    st.session_state.app_state['generated_code'] = generated_code_string
                    # Switch to the 'generated_app' view
                    switch_view('generated_app')
                except Exception as e:
                    st.error(f"❌ An error occurred during generation: {e}")
    else:
        st.info("\U0001F4C1 Please upload a valid task bundle zip to continue")

# ==============================================================================
# VIEW 2: The Dynamically Generated App
# ==============================================================================
elif current_view == "generated_app":
    st.sidebar.button("⬅️ Back to Uploader", on_click=switch_view, args=('uploader',))

    if st.sidebar.button("🔄 Regenerate UI"):
        # Use the path to the current task.yaml
        parent_path = st.session_state.app_state.get('data_path')
        task_yaml_file_to_use = os.path.join(parent_path, 'task.yaml')

        current_code = st.session_state.app_state.get('generated_code', '')
        current_error = st.session_state.app_state.get('last_error', '')
        # print("current error: ", current_error)

        # Update task.yaml with previous_attempt
        import yaml as pyyaml
        if task_yaml_file_to_use and os.path.isfile(task_yaml_file_to_use):
            try:
                with open(task_yaml_file_to_use, 'r', encoding='utf-8') as f:
                    task_yaml_data = pyyaml.safe_load(f)
                if 'previous_attempt' not in task_yaml_data:
                    task_yaml_data['previous_attempt'] = {}
                task_yaml_data['previous_attempt']['code'] = current_code
                task_yaml_data['previous_attempt']['feedback'] = current_error
                with open(task_yaml_file_to_use, 'w', encoding='utf-8') as f:
                    pyyaml.dump(task_yaml_data, f, allow_unicode=True)
            except Exception as e:
                st.warning(f"Could not update previous_attempt in task.yaml: {e}")

        st.info(f"Regenerating from: {parent_path}") # Debug print
        with st.spinner("Regenerating UI... This may take a moment."):
            try:
                # Re-run the generation process
                new_generated_code = main.main(parent_path)
                st.session_state.app_state['generated_code'] = new_generated_code
                st.session_state.app_state['last_error'] = ''
                # Trigger a rerun to display the newly generated UI
                st.session_state['should_rerun'] = True
            except Exception as e:
                st.session_state.app_state['last_error'] = str(e)
                st.error(f"❌ An error occurred during regeneration: {e}")

    st.sidebar.markdown("---")
    
    generated_code = st.session_state.app_state.get('generated_code')
    
    if generated_code:
        st.sidebar.subheader("📄 Generated Code")
        with st.sidebar.expander("Click to view the code running this page"):
            st.code(generated_code, language='python')
        
        try:
            # Prepare a namespace for execution.
            # We pass a dictionary of modules that the generated code might need.
            namespace = {
                "st": st,
                "os": os,
                "yaml": yaml,
                "tempfile": tempfile,
                # Add any other modules the LLM is likely to use
                "pandas": __import__("pandas"),
                "numpy": __import__("numpy"),
                "requests": __import__("requests"),
                "io": io,
                "PIL": __import__("PIL"),
                "librosa": __import__("librosa"),
            }
            # Execute the generated code within the prepared namespace
            exec(generated_code, namespace)
            # If a main() function is defined, call it to ensure UI is rendered
            if 'main' in namespace and callable(namespace['main']):
                namespace['main']()
            
        except Exception as e:
            # Keep track of errors
            print(f"{e}")
            st.session_state.app_state['last_error'] = str(e)
            st.error(f"❌ An error occurred while running the generated code: {e}")
            st.code(generated_code, language='python')
    else:
        st.error("No generated code found.")
        st.button("Go back to uploader", on_click=switch_view, args=('uploader',))

# At the end of the main script, after all UI logic:
if st.session_state.get('should_rerun', False):
    st.session_state['should_rerun'] = False
    print("Triggering rerun...")
    st.rerun()