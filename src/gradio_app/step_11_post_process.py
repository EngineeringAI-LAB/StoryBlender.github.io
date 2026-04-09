import gradio as gr
from .blender_client import BlenderClient


def set_render(blender_client, engine, samples, persistent_data):
    """Set render engine and settings for all scenes.
    
    Args:
        blender_client: BlenderClient instance for communicating with Blender
        engine: Render engine ("EEVEE" or "Cycles")
        samples: Number of render samples (optional)
        persistent_data: Whether to enable persistent data for faster re-renders
        
    Returns:
        dict: Result with success/error and data
    """
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return {
            "error": f"⚠️ {message}"
        }
    
    # Convert samples to int if provided
    samples_int = int(samples) if samples is not None and samples != "" else None
    
    # Call set_render via BlenderClient
    try:
        response = blender_client.set_render(
            engine=engine,
            samples=samples_int,
            persistent_data=persistent_data
        )
        
        if response.get("status") == "error":
            return {
                "error": f"⚠️ Blender error: {response.get('message', 'Unknown error')}"
            }
        
        result = response.get("result", {})
        
        if result.get("success"):
            scenes = result.get("scenes", [])
            scene_count = len(scenes)
            
            # Build detailed message
            details = []
            for scene_info in scenes:
                scene_name = scene_info.get("scene", "Unknown")
                scene_engine = scene_info.get("engine", engine)
                render_samples = scene_info.get("render_samples", "N/A")
                device = scene_info.get("device", "")
                
                detail = f"**{scene_name}**: {scene_engine}"
                if device:
                    detail += f" ({device})"
                detail += f", {render_samples} samples"
                details.append(detail)
            
            details_str = "\n".join([f"- {d}" for d in details]) if details else ""
            
            return {
                "success": True,
                "message": f"✅ Render settings configured for {scene_count} scene(s)!\n\n{details_str}",
            }
        else:
            error_msg = result.get("error", "Unknown error")
            return {
                "error": f"⚠️ {error_msg}"
            }
                
    except Exception as e:
        return {
            "error": f"⚠️ Failed to set render: {str(e)}"
        }


def show_loading_and_set_render(blender_client, engine, samples, persistent_data):
    """Show loading state while setting render."""
    # Loading state
    loading_state = (
        gr.update(value="🔄 **Configuring render settings...** Please wait.", visible=True),
        gr.update(visible=False),  # Hide button
    )
    
    yield loading_state
    
    # Set render
    result = set_render(blender_client, engine, samples, persistent_data)
    
    # Final state
    if result.get("success"):
        success_msg = result.get("message", "✅ Render settings configured!")
        yield (
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True),  # Show button
        )
    else:
        error_msg = result.get("error", "⚠️ Unknown error occurred")
        yield (
            gr.update(value=error_msg, visible=True),
            gr.update(visible=True),  # Show button
        )


def create_set_render_wrapper(blender_client):
    """Factory function to create a set render wrapper bound to specific components."""
    def set_render_wrapper(engine, samples, persistent_data):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_set_render(blender_client, engine, samples, persistent_data):
            yield result
    return set_render_wrapper


def create_post_process_ui(project_dir, blender_client):
    """Create the Step 11: Post Process UI section.
    
    Args:
        project_dir: Gradio component for project directory
        blender_client: BlenderClient instance for communicating with Blender
    
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 11: Post Process")
    
    # ============================================================================
    # Step 11.1: Set Render
    # ============================================================================
    gr.Markdown("### Step 11.1: Set Render")
    gr.Markdown("Configure render engine and settings for all scenes in the Blender file.")
    
    with gr.Row():
        render_engine = gr.Dropdown(
            label="Render Engine",
            choices=["EEVEE", "Cycles"],
            value="Cycles",
            interactive=True,
        )
        render_samples = gr.Number(
            label="Render Samples (optional)",
            value=64,
            precision=0,
            interactive=True,
            info="EEVEE default: 64, Cycles default: 64"
        )
        persistent_data = gr.Checkbox(
            label="Persistent Data",
            value=True,
            interactive=True,
            info="Enable for faster re-renders (keeps data in memory)"
        )
    
    # Status indicator for Step 11.1
    set_render_status = gr.Markdown(value="", visible=False)
    
    # Set Render button
    set_render_btn = gr.Button(
        "🎬 Set Render",
        variant="primary",
        size="lg"
    )
    
    # Create wrapper function for the button
    set_render_wrapper = create_set_render_wrapper(blender_client)
    
    # Set Render button click handler
    set_render_btn.click(
        fn=set_render_wrapper,
        inputs=[render_engine, render_samples, persistent_data],
        outputs=[
            set_render_status,
            set_render_btn,
        ],
    )
    
    return {
        "render_engine": render_engine,
        "render_samples": render_samples,
        "persistent_data": persistent_data,
        "set_render_status": set_render_status,
        "set_render_btn": set_render_btn,
    }
