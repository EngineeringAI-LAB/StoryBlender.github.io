import os
import json
import logging
import gradio as gr
from .path_utils import make_paths_absolute, make_paths_relative

logger = logging.getLogger(__name__)


class JSONEditorComponent:
    """Reusable JSON editor component with file path tracking, editing, and version control."""
    
    def __init__(self, label="JSON", visible_initially=False, save_path=None, file_basename=None, use_version_control=True, language="json", json_root_keys_list=None, title=None):
        """
        Initialize the JSON editor component.
        
        Args:
            label: Label for the code editor
            visible_initially: Whether the editor should be visible initially
            save_path: Directory path where JSON files will be saved (e.g., project_dir/director)
            file_basename: Base name for JSON files (e.g., "director" for director_v1.json)
            use_version_control: If True (default), saves with version suffix (_v1, _v2, etc.).
                                 If False, saves as file_basename.json (overwrites existing).
            language: Language for syntax highlighting (default: "json"). Options: "json", "shell", "python", etc.
            json_root_keys_list: List of root-level keys to display/edit (e.g., ["asset_sheet"]).
                                 If None, the entire JSON is shown. When saving, the full file is
                                 preserved and only the displayed keys are updated.
            title: Title shown in notifications (e.g., "Step 2.2"). If None, uses label.
        """
        self.label = label
        self.title = title or label
        self.visible_initially = visible_initially
        self.save_path = save_path
        self.file_basename = file_basename
        self.use_version_control = use_version_control
        self.language = language
        self.json_root_keys_list = json_root_keys_list
        self.subfolder = None  # Will be set by setup_resume_with_project_dir if needed
        self.project_dir = None  # Set when project_dir is known, for path conversion
        
        # Create components
        self.file_path_display = gr.Markdown(value="", visible=False)
        self.current_filepath = gr.State(value="")
        self.code_editor_wrapper = gr.Column(visible=visible_initially)
        with self.code_editor_wrapper:
            self.code_editor = gr.Code(
                label=label,
                language=language,
                lines=20,
                interactive=True
            )
        
        # Action buttons
        self.resume_btn = gr.Button("📂 Resume Latest", variant="secondary", visible=True)
        self.toggle_view_btn = gr.Button("👁️ Show Editor", variant="secondary", visible=False)
        self.edit_btn = gr.Button("✏️ Edit", variant="secondary", visible=False)
        self.save_btn = gr.Button("💾 Save", variant="primary", visible=False)
        self.copy_btn = gr.Button("📋 Copy to Clipboard", variant="secondary", visible=False)
        
        # State to track code editor visibility
        self.code_editor_visible = gr.State(value=visible_initially)
    
    def setup_handlers(self, project_dir_component):
        """Set up all event handlers for the component.
        
        Must be called after component creation to properly wire up handlers.
        
        Args:
            project_dir_component: Gradio Textbox component for project directory
        """
        # Edit button handler - also shows the code editor if hidden
        self.edit_btn.click(
            fn=lambda fp: (
                gr.update(visible=True),   # Show code_editor
                gr.update(visible=False),  # Hide file_path_display
                gr.update(visible=False),  # Hide edit_btn
                gr.update(visible=True),   # Show save_btn
                True,                      # Update code_editor_visible state
                gr.update(value="👁️ Hide Editor"),  # Sync toggle label
            ),
            inputs=[self.current_filepath],
            outputs=[self.code_editor_wrapper, self.file_path_display, self.edit_btn, self.save_btn, self.code_editor_visible, self.toggle_view_btn],
            queue=False,
            concurrency_limit=None,
            show_progress="hidden",
        )
        
        # Save button handler
        self.save_btn.click(
            fn=self._handle_save,
            inputs=[self.code_editor, self.current_filepath],
            outputs=[self.code_editor, self.file_path_display, self.edit_btn, self.save_btn, self.current_filepath, self.code_editor_visible, self.toggle_view_btn],
            queue=False,
            concurrency_limit=None,
            show_progress="hidden",
        )
        
        # Toggle view button handler
        self.toggle_view_btn.click(
            fn=self._toggle_visibility,
            inputs=[self.code_editor_visible],
            outputs=[self.code_editor_wrapper, self.edit_btn, self.toggle_view_btn, self.code_editor_visible],
            queue=False,
            concurrency_limit=None,
            show_progress="hidden",
        )
        
        # Resume button handler - shows editor briefly to initialize CodeMirror,
        # then .then() hides it so user sees it hidden by default.
        self.resume_btn.click(
            fn=self._handle_resume,
            inputs=[project_dir_component],
            outputs=self._get_resume_outputs(),
            queue=False,
            concurrency_limit=None,
            show_progress="hidden",
        ).then(
            fn=self._post_load_hide,
            inputs=[self.current_filepath],
            outputs=[self.code_editor_wrapper, self.edit_btn, self.toggle_view_btn, self.code_editor_visible],
            queue=False,
            show_progress="hidden",
        )
        
        # Copy button handler - uses JavaScript to copy content to clipboard
        self.copy_btn.click(
            fn=None,
            inputs=[self.code_editor],
            outputs=None,
            js="(text) => { navigator.clipboard.writeText(text); return []; }"
        )
    
    def _post_load_hide(self, filepath):
        """Called by .then() after resume/load to hide editor after CodeMirror has initialized.
        Only hides if a file was actually loaded (filepath is not empty)."""
        if filepath:
            return (
                gr.update(visible=False),                              # code_editor_wrapper: hide
                gr.update(visible=True),                               # edit_btn: show
                gr.update(value="👁️ Show Editor"),                     # toggle_view_btn label
                False,                                                 # code_editor_visible
            )
        # No file loaded - don't change anything
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    
    def _toggle_visibility(self, is_visible):
        """Toggle code editor and edit button visibility."""
        show = not is_visible
        return (
            gr.update(visible=show),
            gr.update(visible=not show),
            gr.update(value="👁️ Hide Editor" if show else "👁️ Show Editor"),
            show,
        )
    
    def _get_resume_outputs(self):
        """Get the output components for resume button."""
        return [
            self.code_editor,
            self.code_editor_wrapper,
            self.file_path_display,
            self.current_filepath,
            self.edit_btn,
            self.save_btn,
            self.resume_btn,
            self.toggle_view_btn,
            self.code_editor_visible,
            self.copy_btn,  # visibility
        ]
    
    def setup_resume_with_project_dir(self, project_dir_component, subfolder=None):
        """
        Setup handlers with project directory component.
        
        Args:
            project_dir_component: Gradio Textbox component for project directory
            subfolder: Optional subfolder within project_dir where files are saved (e.g., "models")
                       If None, defaults to "director" for director files, otherwise project_dir root.
        """
        self.subfolder = subfolder
        self.setup_handlers(project_dir_component)
    
    def _handle_resume(self, project_dir=None):
        """Handle resume button click - load the latest version of JSON file."""
        # Update save_path from project_dir if provided
        if project_dir:
            self.project_dir = project_dir
            if self.subfolder:
                self.save_path = os.path.join(project_dir, self.subfolder)
            else:
                self.save_path = project_dir
        
        latest_path = self.get_path_to_latest_json()
        
        if latest_path:
            json_content = self.load_json_from_file(latest_path)
            file_path_display = f"**Resumed from:** `{latest_path}`"
            gr.Info(title=f"✅ {self.title}",message="JSON load successfully!", duration=10)
            
            return (
                gr.update(value=json_content),  # code_editor: value only
                gr.update(visible=True),        # code_editor_wrapper: show immediately
                gr.update(value=file_path_display, visible=True),
                latest_path,                    # current_filepath
                gr.update(visible=True),        # Show edit_btn (user clicks to view/edit JSON)
                gr.update(visible=False),       # Hide save_btn
                gr.update(visible=False),       # Hide resume_btn
                gr.update(visible=True, value="👁️ Hide Editor"),  # Show toggle_view_btn
                True,                           # Update visibility state (editor is shown)
                gr.update(visible=True),        # Show copy_btn
            )
        else:
            return (
                gr.update(value=""),            # code_editor: clear value
                gr.update(visible=False),       # code_editor_wrapper: hide
                gr.update(value="**No previous version found.** Generate a new script first.", visible=True),
                "",                             # current_filepath
                gr.update(visible=False),       # Keep edit_btn hidden
                gr.update(visible=False),       # Keep save_btn hidden
                gr.update(visible=True),        # Keep resume_btn visible
                gr.update(visible=False, value="👁️ Show Editor"),  # Keep toggle_view_btn hidden
                False,                          # Update visibility state
                gr.update(visible=False),       # Keep copy_btn hidden
            )
    
    def _handle_save(self, code, path):
        """Handle saving edited JSON."""
        result = self._save_edited_json(code, path)
        return (
            result[0],  # code_editor content
            result[1],  # file_path_display content
            result[2],  # edit_btn visibility
            result[3],  # save_btn visibility
            result[4],  # current_filepath
            True,       # code_editor_visible state
            gr.update(value="👁️ Hide Editor"),  # Sync toggle label (editor visible after save)
        )
    
    def _save_edited_json(self, code_content, current_filepath):
        """Save the edited JSON with incremented version number."""
        # Determine save directory and basename
        if current_filepath and os.path.exists(current_filepath):
            directory = os.path.dirname(current_filepath)
            original_filename = os.path.basename(current_filepath)
            # Extract base name from filename
            if "_v" in original_filename:
                base_name = original_filename.split("_v")[0]
            else:
                base_name = original_filename.replace(".json", "")
        elif self.save_path and self.file_basename:
            directory = self.save_path
            base_name = self.file_basename
        else:
            error_msg = "Error: No valid file path or save configuration"
            return (
                gr.update(value=error_msg),
                gr.update(visible=False, value=""),
                gr.update(visible=True),
                gr.update(visible=False),
                current_filepath,
            )
        
        try:
            # Parse the code content to validate JSON
            edited_data = json.loads(code_content)
            
            # If filtering is active, merge edited keys back into the full file
            if self.json_root_keys_list is not None and current_filepath and os.path.exists(current_filepath):
                full_data = self._load_full_json(current_filepath)
                if full_data is not None:
                    json_data = self._merge_filtered_into_full(edited_data, full_data)
                else:
                    json_data = edited_data
            else:
                json_data = edited_data
            
            # Convert paths to relative before saving
            if self.project_dir:
                try:
                    json_data = make_paths_relative(json_data, self.project_dir)
                except Exception as e:
                    logger.warning("JSONEditor: path conversion on save failed: %s", e)
            
            # Get next version path
            output_path, version = self._get_next_version_path(directory, base_name)
            
            # Save the full file
            with open(output_path, "w") as f:
                json.dump(json_data, f, indent=2)
            
            # Display only the filtered content
            display_data = self._filter_json(json_data)
            saved_content = json.dumps(display_data, indent=2)
            file_path_display = f"**File saved:** `{output_path}`"
            
            return (
                gr.update(value=saved_content),
                gr.update(visible=True, value=file_path_display),
                gr.update(visible=True),
                gr.update(visible=False),
                output_path,
            )
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {str(e)}"
            return (
                gr.update(value=code_content + "\n\n# " + error_msg),
                gr.update(visible=False, value=""),
                gr.update(visible=True),
                gr.update(visible=False),
                current_filepath,
            )
        except Exception as e:
            error_msg = f"Error saving file: {str(e)}"
            return (
                gr.update(value=code_content + "\n\n# " + error_msg),
                gr.update(visible=False, value=""),
                gr.update(visible=True),
                gr.update(visible=False),
                current_filepath,
            )
    
    def _filter_json(self, json_data):
        """Filter JSON data to only include the specified root keys."""
        if self.json_root_keys_list is None:
            return json_data
        return {k: json_data[k] for k in self.json_root_keys_list if k in json_data}
    
    def _merge_filtered_into_full(self, filtered_data, full_data):
        """Merge edited filtered keys back into the full JSON data."""
        if self.json_root_keys_list is None:
            return filtered_data
        merged = dict(full_data)
        for k in self.json_root_keys_list:
            if k in filtered_data:
                merged[k] = filtered_data[k]
        return merged
    
    def _load_full_json(self, filepath):
        """Load the full JSON data dict from file (with path conversion)."""
        try:
            if filepath and os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if self.project_dir:
                    try:
                        data = make_paths_absolute(data, self.project_dir)
                    except Exception as e:
                        logger.warning("JSONEditor: path conversion on load failed: %s", e)
                return data
            return None
        except Exception:
            return None
    
    def load_json_from_file(self, filepath):
        """Load JSON content from file and return as formatted string (filtered by json_root_keys_list)."""
        try:
            if filepath and os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                if self.project_dir:
                    try:
                        json_data = make_paths_absolute(json_data, self.project_dir)
                    except Exception as e:
                        logger.warning("JSONEditor: path conversion on load failed: %s", e)
                filtered = self._filter_json(json_data)
                return json.dumps(filtered, indent=2)
            return ""
        except Exception as e:
            return f"Error loading file: {str(e)}"
    
    def set_save_path(self, save_path):
        """Set the save path dynamically (useful when project_dir changes)."""
        self.save_path = save_path
    
    def get_path_to_latest_json(self, save_dir=None, basename=None):
        """
        Get the absolute path to the latest version of the JSON file.
        
        Args:
            save_dir: Directory to look in (uses self.save_path if None)
            basename: Base name for the file (uses self.file_basename if None)
        
        Returns:
            str: Absolute path to the latest version, or None if no file exists
        """
        save_dir = save_dir or self.save_path
        basename = basename or self.file_basename
        
        if not save_dir or not basename:
            return None
        
        if not os.path.exists(save_dir):
            return None
        
        # If not using version control, return the non-versioned file path
        if not self.use_version_control:
            non_versioned_path = os.path.join(save_dir, f"{basename}.json")
            return non_versioned_path if os.path.exists(non_versioned_path) else None
        
        # Find all versioned files
        latest_version = 0
        latest_path = None
        
        for filename in os.listdir(save_dir):
            if filename.startswith(f"{basename}_v") and filename.endswith(".json"):
                try:
                    version_str = filename[len(f"{basename}_v"):-5]  # Remove prefix and .json
                    version = int(version_str)
                    if version > latest_version:
                        latest_version = version
                        latest_path = os.path.join(save_dir, filename)
                except ValueError:
                    continue
        
        return latest_path
    
    def _get_next_version_path(self, save_dir, basename):
        """
        Get the next version file path for saving.
        
        Args:
            save_dir: Directory to save in
            basename: Base name for the file (e.g., "director")
        
        Returns:
            tuple: (filepath, version_number) - version_number is None if not using version control
        """
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # If not using version control, return simple path
        if not self.use_version_control:
            filepath = os.path.join(save_dir, f"{basename}.json")
            return filepath, None
        
        # Find existing versions
        version = 1
        while True:
            filename = f"{basename}_v{version}.json"
            filepath = os.path.join(save_dir, filename)
            if not os.path.exists(filepath):
                return filepath, version
            version += 1
    
    def save_json_data(self, json_data, save_dir=None, basename=None):
        """
        Save JSON data as a new versioned file.
        
        Args:
            json_data: Dictionary to save as JSON
            save_dir: Directory to save in (uses self.save_path if None)
            basename: Base name for the file (uses self.file_basename if None)
        
        Returns:
            str: Path to the saved file, or None on error
        """
        save_dir = save_dir or self.save_path
        basename = basename or self.file_basename
        
        if not save_dir or not basename:
            print("Error: save_path and file_basename must be set")
            return None
        
        try:
            # Convert paths to relative before saving
            save_data = json_data
            if self.project_dir:
                try:
                    save_data = make_paths_relative(json_data, self.project_dir)
                except Exception as e:
                    logger.warning("JSONEditor: path conversion on save failed: %s", e)
            
            filepath, version = self._get_next_version_path(save_dir, basename)
            with open(filepath, "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"Saved JSON to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving JSON: {str(e)}")
            return None
    
    def update_with_result(self, result):
        """
        Update the component with generation result.
        Returns tuple of outputs for: code_editor, code_editor_wrapper, file_path_display,
        current_filepath, edit_btn, save_btn, resume_btn, toggle_view_btn, code_editor_visible, copy_btn
        """
        if result and "output_path" in result:
            file_path = result["output_path"]
            file_path_display = f"**File saved:** `{file_path}`"
            json_content = self.load_json_from_file(file_path)
            gr.Info(title=f"✅ {self.title}", message="Task finished!", duration=0)
            
            return (
                gr.update(value=json_content),  # code_editor: value only
                gr.update(visible=False),       # code_editor_wrapper: hidden by default
                gr.update(value=file_path_display, visible=True),
                file_path,                      # current_filepath
                gr.update(visible=True),        # Show edit_btn
                gr.update(visible=False),       # Hide save_btn
                gr.update(visible=False),       # Hide resume_btn
                gr.update(visible=True, value="👁️ Show Editor"),  # Show toggle_view_btn
                False,                          # Editor is hidden
                gr.update(visible=True),        # Show copy_btn
            )
        elif result is None:
            # Loading state - show wrapper and toggle briefly to initialize DOM/CodeMirror.
            # The final generator yield will update these to their correct states.
            return (
                gr.update(value=""),            # code_editor: clear value
                gr.update(visible=True),        # code_editor_wrapper: show briefly for DOM init
                gr.update(value="", visible=False),
                "",                             # current_filepath
                gr.update(visible=False),       # Keep edit_btn hidden
                gr.update(visible=False),       # Keep save_btn hidden
                gr.update(visible=False),       # Hide resume_btn
                gr.update(visible=True, value="👁️ Show Editor"),  # Show toggle for DOM init
                True,                           # Temporarily visible
                gr.update(visible=False),       # Keep copy_btn hidden
            )
        else:
            # Error or no output_path
            return (
                gr.update(value=""),            # code_editor: clear value
                gr.update(visible=False),       # code_editor_wrapper: hide
                gr.update(value="", visible=False),
                "",                             # current_filepath
                gr.update(visible=False),       # Keep edit_btn hidden
                gr.update(visible=False),       # Keep save_btn hidden
                gr.update(visible=True),        # Keep resume_btn visible
                gr.update(visible=False, value="👁️ Show Editor"),  # Keep toggle_view_btn hidden
                False,                          # Update visibility state
                gr.update(visible=False),       # Keep copy_btn hidden
            )
    
    def get_output_components(self):
        """Get list of output components in order for event handlers."""
        return [
            self.code_editor,
            self.code_editor_wrapper,
            self.file_path_display,
            self.current_filepath,
            self.edit_btn,
            self.save_btn,
            self.resume_btn,
            self.toggle_view_btn,
            self.code_editor_visible,
            self.copy_btn,  # for visibility update
        ]
