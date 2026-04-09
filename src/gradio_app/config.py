"""Configuration UI components and handlers for StoryBlender."""

import os
import gradio as gr


def create_config_ui():
    """Create configuration UI components.
    
    Returns:
        A dictionary containing all configuration components.
    """
    gr.Markdown("## Configuration")
    
    with gr.Row():
        gemini_image_model = gr.Textbox(
            label="Gemini Image Model",
            value="gemini-3.1-flash-image-preview",
            info="The Gemini image model used during text to image to 3D",
            visible=True
        )
        gemini_api_key = gr.Textbox(
            label="Gemini API Key",
            value="",
            type="password",
            info="Your Gemini API key for authentication",
            visible=True
        )
        gemini_api_base = gr.Textbox(
            label="Gemini API Base",
            value="",
            info="Custom API base URL for Gemini (leave empty for default)",
            visible=True
        )
    
    with gr.Row():
        # Reasoning Model
        reasoning_model = gr.Textbox(
            label="Reasoning Model",
            value="gemini-3.1-pro-preview",
            info="The primary reasoning model for complex reasoning tasks",
            visible=True
        )
        vision_model = gr.Textbox(
            label="Vision Model",
            value="gemini-3-flash-preview",
            info="A lighter model for fast multi-model inference",
            visible=True
        )
        anyllm_api_key = gr.Textbox(
            label="AnyLLM API Key",
            value="",
            type="password",
            info="Your AnyLLM API key",
            visible=True
        )
        anyllm_api_base = gr.Textbox(
            label="AnyLLM API Base",
            value="",
            info="API base URL for AnyLLM",
            visible=True
        )
        anyllm_provider = gr.Textbox(
            label="AnyLLM Provider",
            value="gemini",
            info="LLM provider (default: gemini)",
            visible=True
        )
    
    with gr.Row():
        sketchfab_api_key = gr.Textbox(
            label="Sketchfab API Key",
            value="",
            type="password",
            info="Your Sketchfab API key for 3D model retrieval",
            visible=True
        )
        
    with gr.Row():
        meshy_api_key = gr.Textbox(
            label="Meshy API Key",
            value="",
            type="password",
            info="Your Meshy API key for 3D model generation",
            visible=True
        )
        meshy_model = gr.Textbox(
                label="Meshy Model",
                value="latest",
                info="Meshy AI model version to use for 3D generation (default: 'latest')",
                visible=True
        )
    
    with gr.Row():
        tencent_secret_id = gr.Textbox(
            label="Tencent Cloud Secret ID",
            value="",
            type="password",
            info="Your Tencent Cloud Secret ID for Hunyuan3D",
            visible=True
        )
        tencent_secret_key = gr.Textbox(
            label="Tencent Cloud Secret Key",
            value="",
            type="password",
            info="Your Tencent Cloud Secret Key for Hunyuan3D",
            visible=True
        )
        ai_platform = gr.Dropdown(
            label="AI Platform",
            choices=["Hunyuan3D", "Meshy"],
            value="Meshy",
            info="AI platform for 3D model generation",
            visible=True
        )
    
    with gr.Row():
        project_dir = gr.Textbox(
            label="Project Absolute Directory",
            value="",
            placeholder="/Users/username/projects/my_project",
            info="⚠️ Must be an absolute path to a directory where the generated files will be saved",
            visible=True
        )
    
    save_config_btn = gr.Button("💾 Save Configuration", variant="secondary")
    edit_config_btn = gr.Button("⚙️ Edit Configuration", variant="secondary", visible=False)
    config_warning = gr.Markdown("", visible=False)
    
    return {
        "gemini_image_model": gemini_image_model,
        "gemini_api_key": gemini_api_key,
        "gemini_api_base": gemini_api_base,
        "reasoning_model": reasoning_model,
        "vision_model": vision_model,
        "anyllm_api_key": anyllm_api_key,
        "anyllm_api_base": anyllm_api_base,
        "anyllm_provider": anyllm_provider,
        "sketchfab_api_key": sketchfab_api_key,
        "meshy_api_key": meshy_api_key,
        "meshy_model": meshy_model,
        "tencent_secret_id": tencent_secret_id,
        "tencent_secret_key": tencent_secret_key,
        "ai_platform": ai_platform,
        "project_dir": project_dir,
        "save_config_btn": save_config_btn,
        "edit_config_btn": edit_config_btn,
        "config_warning": config_warning,
    }


def setup_config_handlers(config_components):
    """Setup click handlers for configuration buttons.
    
    Args:
        config_components: Dictionary of config components from create_config_ui()
    """
    gemini_image_model = config_components["gemini_image_model"]
    gemini_api_key = config_components["gemini_api_key"]
    gemini_api_base = config_components["gemini_api_base"]
    reasoning_model = config_components["reasoning_model"]
    vision_model = config_components["vision_model"]
    anyllm_api_key = config_components["anyllm_api_key"]
    anyllm_api_base = config_components["anyllm_api_base"]
    anyllm_provider = config_components["anyllm_provider"]
    sketchfab_api_key = config_components["sketchfab_api_key"]
    meshy_api_key = config_components["meshy_api_key"]
    meshy_model = config_components["meshy_model"]
    tencent_secret_id = config_components["tencent_secret_id"]
    tencent_secret_key = config_components["tencent_secret_key"]
    ai_platform = config_components["ai_platform"]
    project_dir = config_components["project_dir"]
    save_config_btn = config_components["save_config_btn"]
    edit_config_btn = config_components["edit_config_btn"]
    config_warning = config_components["config_warning"]
    
    def validate_and_save(project_dir_value):
        """Validate project_dir and save configuration if valid."""
        # Strip wrapping single quotes (macOS), backticks, or double quotes (Windows) from path
        if project_dir_value:
            project_dir_value = project_dir_value.strip()
            if (project_dir_value.startswith("'") and project_dir_value.endswith("'")) or \
               (project_dir_value.startswith('`') and project_dir_value.endswith('`')) or \
               (project_dir_value.startswith('"') and project_dir_value.endswith('"')):
                project_dir_value = project_dir_value[1:-1]
        # Check if project_dir is empty or not an absolute path
        if not project_dir_value or not project_dir_value.strip():
            return (
                gr.update(),  # gemini_image_model
                gr.update(),  # reasoning_model
                gr.update(),  # vision_model
                gr.update(),  # gemini_api_key
                gr.update(),  # gemini_api_base
                gr.update(),  # anyllm_api_key
                gr.update(),  # anyllm_api_base
                gr.update(),  # anyllm_provider
                gr.update(),  # sketchfab_api_key
                gr.update(),  # meshy_api_key
                gr.update(),  # meshy_model
                gr.update(),  # tencent_secret_id
                gr.update(),  # tencent_secret_key
                gr.update(),  # ai_platform
                gr.update(),  # project_dir
                gr.update(),  # save_config_btn
                gr.update(),  # edit_config_btn
                gr.update(value="⚠️ **Warning:** Project Directory cannot be empty. Please provide a valid absolute path.", visible=True),  # config_warning
            )
        
        if not os.path.isabs(project_dir_value.strip()):
            return (
                gr.update(),  # gemini_image_model
                gr.update(),  # reasoning_model
                gr.update(),  # vision_model
                gr.update(),  # gemini_api_key
                gr.update(),  # gemini_api_base
                gr.update(),  # anyllm_api_key
                gr.update(),  # anyllm_api_base
                gr.update(),  # anyllm_provider
                gr.update(),  # sketchfab_api_key
                gr.update(),  # meshy_api_key
                gr.update(),  # meshy_model
                gr.update(),  # tencent_secret_id
                gr.update(),  # tencent_secret_key
                gr.update(),  # ai_platform
                gr.update(),  # project_dir
                gr.update(),  # save_config_btn
                gr.update(),  # edit_config_btn
                gr.update(value=f"⚠️ **Warning:** '{project_dir_value}' is not an absolute path. Please provide a path starting with '/'.", visible=True),  # config_warning
            )
        
        # Add project_dir to Gradio's static paths so files can be served
        # This allows the app to serve files from the working directory after launch
        project_path = project_dir_value.strip()
        gr.set_static_paths(paths=[project_path])
        
        # Validation passed, proceed with saving
        return (
            gr.update(visible=False),  # gemini_image_model
            gr.update(visible=False),  # reasoning_model
            gr.update(visible=False),  # vision_model
            gr.update(visible=False),  # gemini_api_key
            gr.update(visible=False),  # gemini_api_base
            gr.update(visible=False),  # anyllm_api_key
            gr.update(visible=False),  # anyllm_api_base
            gr.update(visible=False),  # anyllm_provider
            gr.update(visible=False),  # sketchfab_api_key
            gr.update(visible=False),  # meshy_api_key
            gr.update(visible=False),  # meshy_model
            gr.update(visible=False),  # tencent_secret_id
            gr.update(visible=False),  # tencent_secret_key
            gr.update(visible=False),  # ai_platform
            gr.update(value=project_path, visible=False),  # project_dir
            gr.update(visible=False),  # save_config_btn
            gr.update(visible=True),   # edit_config_btn
            gr.update(visible=False),  # config_warning - hide warning on success
        )
    
    # Save Configuration button click handler
    save_config_btn.click(
        fn=validate_and_save,
        inputs=[project_dir],
        outputs=[
            gemini_image_model, reasoning_model, vision_model,
            gemini_api_key, gemini_api_base,
            anyllm_api_key, anyllm_api_base, anyllm_provider,
            sketchfab_api_key, meshy_api_key, meshy_model,
            tencent_secret_id, tencent_secret_key, ai_platform,
            project_dir, save_config_btn, edit_config_btn, config_warning
        ],
        concurrency_limit=None,
        show_progress="hidden",
    )
    
    # Edit Configuration button click handler
    edit_config_btn.click(
        fn=lambda: (
            gr.update(visible=True),   # gemini_image_model
            gr.update(visible=True),   # reasoning_model
            gr.update(visible=True),   # vision_model
            gr.update(visible=True),   # gemini_api_key
            gr.update(visible=True),   # gemini_api_base
            gr.update(visible=True),   # anyllm_api_key
            gr.update(visible=True),   # anyllm_api_base
            gr.update(visible=True),   # anyllm_provider
            gr.update(visible=True),   # sketchfab_api_key
            gr.update(visible=True),   # meshy_api_key
            gr.update(visible=True),   # meshy_model
            gr.update(visible=True),   # tencent_secret_id
            gr.update(visible=True),   # tencent_secret_key
            gr.update(visible=True),   # ai_platform
            gr.update(visible=True),   # project_dir
            gr.update(visible=True),   # save_config_btn
            gr.update(visible=False),  # edit_config_btn
            gr.update(visible=False),  # config_warning - hide warning when editing
        ),
        inputs=[],
        outputs=[
            gemini_image_model, reasoning_model, vision_model,
            gemini_api_key, gemini_api_base,
            anyllm_api_key, anyllm_api_base, anyllm_provider,
            sketchfab_api_key, meshy_api_key, meshy_model,
            tencent_secret_id, tencent_secret_key, ai_platform,
            project_dir, save_config_btn, edit_config_btn, config_warning
        ],
        concurrency_limit=None,
        show_progress="hidden",
    )
