import streamlit as st
import asyncio
import json
import os
from datetime import datetime
import time
import base64
from pathlib import Path

try:
    from reddit_agent import RedditResearchAgent
except ImportError:
    st.error("Failed to import RedditResearchAgent. Please ensure the module is installed and properly configured.")
    st.stop()

# Try to import streamlit-pdf-viewer
try:
    from streamlit_pdf_viewer import pdf_viewer
    PDF_VIEWER_AVAILABLE = True
except ImportError:
    PDF_VIEWER_AVAILABLE = False
    st.warning("⚠️ streamlit-pdf-viewer not installed. Run: `pip install streamlit-pdf-viewer` to enable PDF viewing in the app.")

# Page configuration
st.set_page_config(
    page_title="Reddit Pain Point Research AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.0rem;
    }
    .main-header p {
        color: white;
        text-align: center;
        margin: 0.3rem 0 0 0;
        font-size: 1.2rem;
    }
    .step-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bee5eb;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔍 Reddit Pain Point Research AI</h1>
    <p>Discover user pain points and generate solutions from Reddit discussions</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'workflow_running' not in st.session_state:
    st.session_state.workflow_running = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = ""
if 'workflow_complete' not in st.session_state:
    st.session_state.workflow_complete = False
if 'project_id' not in st.session_state:
    st.session_state.project_id = None
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Workflow parameters
    st.subheader("Search Parameters")

    num_keywords = st.slider(
        "Number of keywords to generate",
        min_value=2,
        max_value=8,
        value=5,
        help="AI will generate this many keywords based on your project idea"
    )

    posts_per_subreddit = st.slider(
        "Posts per subreddit",
        min_value=2,
        max_value=10,
        value=5,
        help="How many posts to search in each relevant subreddit"
    )
    
    min_post_score = st.number_input(
        "Minimum post score",
        min_value=0,
        max_value=1000,
        value=5,
        help="Only consider posts with at least this many score"
    )
    
    min_comments = st.number_input(
        "Minimum comments per post",
        min_value=1,
        max_value=100,
        value=2,
        help="Only analyze posts with at least this many comments"
    )
    
    comments_per_post = st.slider(
        "Comments to extract per post",
        min_value=1,
        max_value=50,
        value=5,
        help="How many top comments to analyze from each post"
    )
    
    min_comment_score = st.number_input(
        "Minimum comment score",
        min_value=0,
        max_value=100,
        value=2,
        help="Only consider comments with at least this many upvotes"
    )
    
    st.markdown("---")

    # Project info
    st.subheader("📊 Project Info")
    if st.session_state.project_id:
        st.success(f"Project ID: {st.session_state.project_id}")
    
    if st.session_state.workflow_complete:
        st.success("✅ Workflow Complete!")


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🚀 Project Idea")
    project_idea = st.text_area(
        "Describe your project idea",
        height=450,
        placeholder="Example: We are building an AI workflow to get user pain points from Reddit...",
        help="Provide detailed information about your project. The AI will generate relevant keywords and search for related pain points on Reddit."
    )

    if st.session_state.workflow_running:
        # Show "Stop Research" button when workflow is running
        if st.button("🛑 Stop Research", type="secondary", use_container_width=True):
            st.session_state.workflow_running = False
            st.session_state.current_step = "Stopped by user"
            st.rerun()
    else:
        # Show "Start Research" button when ready to start or after completion
        if st.button("🔍 Start Research", type="primary", use_container_width=True):
            if project_idea.strip():
                st.session_state.workflow_running = True
                st.session_state.workflow_complete = False
                st.session_state.current_step = "Initializing..."
                st.rerun()
            else:
                st.error("Please provide a project idea before starting the research.")

with col2:
    st.subheader("📋 Research Steps")
    steps = [
        "🔤 Generate Keywords",
        "📱 Find Subreddits", 
        "🔍 Search Posts",
        "🤖 Filter with AI",
        "💬 Extract Comments",
        "📊 Analyze Content",
        "🗄️ Store Vectors",
        "📝 Summarize Pain Points",
        "🎯 Generate Solution Keywords",
        "💡 Generate Solutions",
        "📄 Generate Report"
    ]

    for i, step in enumerate(steps, 1):
        step_name = step.split(" ", 1)[1]
        st.markdown(f"{i}. {step}")

# Progress tracking
if st.session_state.workflow_running:
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate workflow execution
    async def run_workflow():
        try:
            projects_path = "projects"
            os.makedirs(projects_path, exist_ok=True)
            queries_db = os.path.join(projects_path, "queries.json")

            # Handle project storage
            if not os.path.exists(queries_db):
                with open(queries_db, "w") as f:
                    project_id = 1
                    projects = [{
                        "id": project_id,
                        "project_idea": project_idea,
                        "created_at": datetime.now().isoformat()
                    }]
                    json.dump(projects, f, indent=2)
            else:
                with open(queries_db, "r") as f:
                    projects = json.load(f)
                is_project_idea_exists = any(project["project_idea"] == project_idea for project in projects)
                if not is_project_idea_exists:
                    project_id = projects[-1]["id"] + 1
                    projects.append({
                        "id": project_id,
                        "project_idea": project_idea,
                        "created_at": datetime.now().isoformat()
                    })
                else:
                    project_id = next(project["id"] for project in projects if project["project_idea"] == project_idea)
                
                with open(queries_db, "w") as f:
                    json.dump(projects, f, indent=2)
            
            st.session_state.project_id = project_id

            config = {
                'num_keywords': num_keywords,
                'posts_per_subreddit': posts_per_subreddit,
                'min_post_score': min_post_score,
                'min_comments': min_comments,
                'comments_per_post': comments_per_post,
                'min_comment_score': min_comment_score
            }

            def update_progress(step_name, details=None):
                if st.session_state.workflow_running:
                    st.session_state.current_step = step_name
                    step_index = next(i for i, s in enumerate(steps) if step_name in s)
                    progress_bar.progress((step_index + 1) / len(steps))
                    status_text.text(f"Processing: {step_name} - {details}")

            # Run the research agent
            agent = RedditResearchAgent(project_id, projects_path, update_progress)
            result =await agent.run_research(project_id, project_idea, config)
            print(result)
            pdf_path = result.get('final_state', {}).get('report_path')
            print(f"pdf_path: {pdf_path}")
            if os.path.exists(pdf_path):
                st.session_state.pdf_path = pdf_path
            st.session_state.workflow_complete = True
            st.session_state.workflow_running = False
            st.session_state.current_step = "Complete!"

            return result
        
        except Exception as e:
            st.error(f"Error running workflow: {str(e)}")
            st.session_state.workflow_running = False
            st.session_state.current_step = f"Error: {str(e)}"
            return None

     # Run the workflow
    if st.session_state.workflow_running:
        result = asyncio.run(run_workflow())

# Results section
if st.session_state.workflow_complete:
    st.markdown("---")
    st.subheader("📊 Research Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box" style="height: 120px; display: flex; align-items: center; justify-content: center;">
            <div>
                <h4>✅ Research Complete!</h4>
                <p>Your Reddit pain point research has been completed successfully.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
            with open(st.session_state.pdf_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data,
                file_name=f"research_report_{st.session_state.project_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    # Display PDF in the app
    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        st.subheader("📄 Generated Report")
        
        if PDF_VIEWER_AVAILABLE:
            try:
                pdf_viewer(
                    st.session_state.pdf_path,
                    width="100%",
                    height=1000,
                    render_text=True,
                    zoom_level="auto",
                    viewer_align="left",
                    show_page_separator=True
                )
            except Exception as e:
                st.error(f"Error displaying PDF: {str(e)}")
                st.info("You can still download the PDF using the button above.")
        else:
            # Fallback: Show installation instructions and basic info
            st.info("📋 To view the PDF report directly in the app, please install the PDF viewer component:")
            st.code("pip install streamlit-pdf-viewer", language="bash")
            st.info("After installation, restart the app to enable PDF viewing.")
            
            # Alternative: Show PDF file info
            try:
                file_size = os.path.getsize(st.session_state.pdf_path)
                st.write(f"📄 **Report File:** {os.path.basename(st.session_state.pdf_path)}")
                st.write(f"📦 **File Size:** {file_size:,} bytes")
                st.write(f"📅 **Generated:** {datetime.fromtimestamp(os.path.getmtime(st.session_state.pdf_path)).strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                st.error(f"Error reading PDF file info: {str(e)}")

    # Option to start new research
    if st.button("🔄 Start New Research", use_container_width=True):
        st.session_state.workflow_running = False
        st.session_state.workflow_complete = False
        st.session_state.current_step = ""
        st.session_state.project_id = None
        st.session_state.pdf_path = None
        st.rerun()
        
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🤖 Powered by AI • Built with Streamlit • Reddit Pain Point Research</p>
</div>
""", unsafe_allow_html=True)