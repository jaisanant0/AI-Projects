from langgraph.graph import StateGraph, START, END
import json
import logging
from typing import Dict, Any
import asyncio
from llm_manager import LLMManager
from reddit_manager import RedditAPIManager 
from vector_manager import VectorDBManager
from report_manager import ReportGenerator

import os
import hashlib
from datetime import datetime

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import random

from dotenv import load_dotenv

load_dotenv()

from json_schemas import ResearchState, PainPoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Suppress HTTP logs from common libraries
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)
logging.getLogger('asyncpraw').setLevel(logging.ERROR)
logging.getLogger('prawcore').setLevel(logging.ERROR)
logging.getLogger('aiohttp').setLevel(logging.ERROR)
logging.getLogger('praw').setLevel(logging.ERROR)
# If you're using weasyprint
logging.getLogger('weasyprint').setLevel(logging.ERROR)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)



class RedditResearchAgent:
    """Main agent orchestrating the research workflow"""

    def __init__(self, project_id: str, projects_path: str, progress_callback=None):
        self.progress_callback = progress_callback 
        self.reddit_manager = RedditAPIManager()
        self.llm_manager = LLMManager(
            model_name=os.getenv("MODEL_NAME"),
            endpoint=os.getenv("ENDPOINT")
        )

        self.projects_path = projects_path
        self.project_id = project_id

        self.vector_db = VectorDBManager()
        self.report_manager = ReportGenerator(project_id=project_id, projects_path=projects_path, vector_db=self.vector_db)
        
        self.workflow = self._build_workflow()

    def _build_workflow(self): 
        """Build the LangGraph workflow"""

        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("generate_keywords", self._generate_keywords)
        workflow.add_node("get_subreddits", self._get_subreddits)
        workflow.add_node("search_subreddits", self._search_subreddits)
        workflow.add_node("llm_filter_posts", self._llm_filter_posts)
        workflow.add_node("extract_comments", self._extract_comments)
        workflow.add_node("analyze_content", self._analyze_content)
        workflow.add_node("store_vectors", self._store_vectors)
        workflow.add_node("summarize_pain_points", self._summarize_pain_points)
        workflow.add_node("generate_solutions_keywords", self.generate_solutions_keywords)
        workflow.add_node("generate_solutions", self.generate_solutions)
        workflow.add_node("generate_report", self._generate_report)


        # Add edges
        workflow.add_edge(START, "generate_keywords")
        workflow.add_edge("generate_keywords", "get_subreddits")
        workflow.add_edge("get_subreddits", "search_subreddits")
        workflow.add_edge("search_subreddits", "llm_filter_posts")
        workflow.add_edge("llm_filter_posts", "extract_comments")
        workflow.add_edge("extract_comments", "analyze_content")
        workflow.add_edge("analyze_content", "store_vectors")
        workflow.add_edge("store_vectors", "summarize_pain_points")
        workflow.add_edge("summarize_pain_points", "generate_solutions_keywords")
        workflow.add_edge("generate_solutions_keywords", "generate_solutions")
        workflow.add_edge("generate_solutions", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile() 

    def _update_progress(self, step_name, details=None):
        """Helper method to update progress"""
        if self.progress_callback:
            self.progress_callback(step_name, details)
    
    async def _generate_keywords(self, state: ResearchState) -> ResearchState:
        """Generate keywords from project idea"""
        logger.info("Generating keywords...") 
        self._update_progress("Generate Keywords", "Starting keyword generation...")

        keywords = await self.llm_manager.generate_keywords(state.project_idea, is_thinking=True)
        
        logger.info(f"Keywords Found: {keywords}")
        self._update_progress("Generate Keywords", f"Generated {len(keywords)} keywords")

        state.keywords = keywords["keywords"][:state.config["num_keywords"]]
        #logger.info(f"Selected Keywords: {state.keywords}")
        return state

    async def _get_subreddits(self, state: ResearchState) -> ResearchState:
        """Get subreddits from keywords"""
        logger.info("Finding subreddits...")
        self._update_progress("Find Subreddits", "Searching for relevant subreddits...")

        subreddits = await self.reddit_manager.get_subreddits(state.keywords)
        unique_subreddits = list(set(subreddits))
        logger.info(f"Found {len(unique_subreddits)} subreddits")
        logger.info(f"Subreddits: {unique_subreddits}")
        self._update_progress("Find Subreddits", f"Found {len(unique_subreddits)} subreddits")

        state.subreddits = unique_subreddits
        return state

    async def _search_subreddits(self, state: ResearchState) -> ResearchState:
        """Search Reddit for relevant posts"""
        
        all_posts = []
        for keyword in state.keywords:
            logger.info(f"Searching Subreddits for keyword: {keyword}")
            self._update_progress("Search Posts", f"Searching Subreddits for keyword: {keyword}")

            posts = await self.reddit_manager.search_subreddits(state.subreddits, 
                                                                keyword, 
                                                                limit=state.config["posts_per_subreddit"],
                                                                min_post_score=state.config["min_post_score"],
                                                                num_comments=state.config["min_comments"])
            all_posts.extend(posts) 
        
        # Remove duplicates
        logger.info(f"Found {len(all_posts)} posts")
        logger.info(f"Removing duplicates...")
        
        unique_posts = {post.id: post for post in all_posts}
        state.reddit_posts = list(unique_posts.values())

        logger.info(f"Found {len(state.reddit_posts)} unique posts")
        self._update_progress("Search Posts", f"Found {len(state.reddit_posts)} posts total")
        return state

    async def _llm_filter_posts(self, state: ResearchState) -> ResearchState: 
        """Filter posts using LLM""" 
        logger.info(f"Filtering posts with AI...") 
        self._update_progress("Filter with AI", "Filtering posts with AI...")

        filtered_posts = await self.llm_manager.filter_posts(state.project_idea, 
                                                             state.reddit_posts)
        
        logger.info(f"Found {len(filtered_posts)} filtered posts")
        self._update_progress("Filter with AI", f"Found {len(filtered_posts)} filtered posts")

        state.filtered_posts = filtered_posts
        return state

    async def _extract_comments(self, state: ResearchState) -> ResearchState:
        """Extract comments from Reddit posts"""
        
        all_comments = []
        for post in state.filtered_posts:
            logger.info(f"Extracting comments for post: {post.title}")
            self._update_progress("Extract Comments", f"Extracting comments for post: {post.title}")
            comments = await self.reddit_manager.get_post_comments(
                post_id=post.id, 
                limit=state.config["comments_per_post"], 
                min_comment_score=state.config["min_comment_score"])
            all_comments.extend(comments)
        
        logger.info(f"Extracted {len(all_comments)} comments")
        self._update_progress("Extract Comments", f"Extracted {len(all_comments)} comments")

        state.reddit_comments = all_comments
        return state

    async def _analyze_content(self, state: ResearchState) -> ResearchState:
        """Analyze content for pain points"""
        logger.info("Analyzing content for pain points...")
        self._update_progress("Analyze Content", "Analyzing content for pain points...")

        pain_points = []
        for post in state.filtered_posts:
            #get comments for the post
            post_comments = []
            for comment in state.reddit_comments:
                if comment.post_id == post.id:
                    post_comments.append(comment)

            extracted_pain_points = await self.llm_manager.extract_pain_points(
                state.project_idea, 
                post,
                post_comments,
                is_thinking=False)
        
            pain_points_texts = extracted_pain_points["pain_points"] 
            for i, text in enumerate(pain_points_texts):
                pain_point_id = hashlib.md5(text.encode()).hexdigest()  
                pain_point_category = await self.llm_manager.categorize_pain_point(state.project_idea, text)   
                pain_point = PainPoint(
                    id=str(pain_point_id),
                    content=text,
                    category=pain_point_category["category"],
                    sources_post=post.id
                )
                pain_points.append(pain_point)

        logger.info(f"Total pain points: {len(pain_points)}")
        self._update_progress("Analyze Content", "Content analysis complete")
        
        state.pain_points = pain_points
        return state

    async def _store_vectors(self, state: ResearchState) -> ResearchState:
        """Store pain points in vector database"""
        logger.info("Creating and storing vector embeddings...")
        self._update_progress("Store Vectors", "Creating and storing vector embeddings...")

        self.vector_db.store_vectors(state.pain_points, state.project_id)

        self._update_progress("Store Vectors", "Vector embeddings created and stored") 

        return state

    async def _summarize_pain_points(self, state: ResearchState) -> ResearchState:
        """Summarize pain points"""
        logger.info("Identifying and summarizing pain points...")
        self._update_progress("Summarize Pain Points", "Identifying and summarizing pain points...")

        unique_pain_points = self.vector_db.get_unique_pain_points(
            project_id=state.project_id
        )

        pain_points = "" 
        for i,pain_point in enumerate(unique_pain_points):
            pain_points += f"{i+1}. {pain_point['content']}\n" 

        summarized_pain_points = await self.llm_manager.summarize_pain_points(state.project_idea, pain_points)
        state.summarized_pain_points = summarized_pain_points

        self._update_progress("Summarize Pain Points", "Pain points summarized")
        return state

    async def generate_solutions_keywords(self, state: ResearchState) -> ResearchState: 
        """Generate solutions for pain points""" 
        logger.info("Generating keywords for solutions...")
        self._update_progress("Generate Solution Keywords", "Generating keywords for solutions...")

        #logger.info(f"Summarized pain points: {state.summarized_pain_points}")
        solution_keywords = await self.llm_manager.generate_solutions_keywords(state.project_idea, 
                                                              state.summarized_pain_points)
        logger.info(f"Solutions keywords: {solution_keywords}")
        self._update_progress("Generate Solution Keywords", f"Generated {len(solution_keywords)} solution keywords")

        state.solution_keywords = solution_keywords["keywords"][:state.config["num_keywords"]]
        return state

    async def generate_solutions(self, state: ResearchState) -> ResearchState: 
        """Generate solutions for pain points""" 
        logger.info("Generating solutions for pain points...")
        self._update_progress("Generate Solutions", "Generating solutions for pain points...")

        solution_subreddits = await self.reddit_manager.get_subreddits(state.solution_keywords)
        logger.info(f"Found {len(solution_subreddits)} solution subreddits") 
        self._update_progress("Generate Solutions", f"Found {len(solution_subreddits)} solution subreddits")

        all_posts = []
        for keyword in state.solution_keywords:
            logger.info(f"Searching Subreddits for Solution keyword: {keyword}")
            self._update_progress("Generate Solutions", f"Searching Subreddits for Solution keyword: {keyword}")
            posts = await self.reddit_manager.search_subreddits(solution_subreddits, 
                                                                keyword, 
                                                                limit=state.config["posts_per_subreddit"],
                                                                min_post_score=state.config["min_post_score"],
                                                                num_comments=state.config["min_comments"])
            all_posts.extend(posts)
        
        logger.info(f"Found {len(all_posts)} Solution posts")
        logger.info(f"Removing duplicates...")

        unique_posts = {post.id: post for post in all_posts}
        state.solution_reddit_posts = list(unique_posts.values())

        logger.info(f"Found {len(state.solution_reddit_posts)} unique posts")
        self._update_progress("Generate Solutions", f"Found {len(state.solution_reddit_posts)} unique posts")
        
        logger.info(f"Filtering solution posts...")
        self._update_progress("Generate Solutions", "Filtering solution posts...")

        summarized_pain_points = state.summarized_pain_points.summarized_pain_points
        all_summarized_pain_points = ""
        for summ_pain_point in summarized_pain_points:
            all_summarized_pain_points += f"{summ_pain_point.theme_name}: {summ_pain_point.description}\n"


        reddit_filtered_posts, reddit_solutions = await self.llm_manager.filter_solution_posts(state.project_idea, 
                                                                               state.solution_reddit_posts,
                                                                               all_summarized_pain_points,
                                                                               filter_threshold=7)
        
        logger.info(f"Found {len(reddit_filtered_posts)} solution filtered posts")
        logger.info(f"Reddit solutions length : {sum(len(s) for s in reddit_solutions)}")
        self._update_progress("Generate Solutions", f"Found {len(reddit_filtered_posts)} solution filtered posts")
        # Check if reddit solutions exceed 10000 words and truncate smartly if needed
        total_reddit_words = sum(len(s.split()) for s in reddit_solutions)
        if total_reddit_words > 10000:
            logger.info(f"Reddit solutions ({total_reddit_words} words) exceed 10000 words. Truncating smartly...")
            truncated_solutions = []
            current_word_count = 0
            word_limit = 10000
            
            for solution in reddit_solutions:
                solution_words = solution.split()
                if current_word_count + len(solution_words) <= word_limit:
                    # Add complete solution if it fits
                    truncated_solutions.append(solution)
                    current_word_count += len(solution_words)
                else:
                    # Add partial solution to reach the limit
                    remaining_words = word_limit - current_word_count
                    if remaining_words > 0:
                        partial_solution = ' '.join(solution_words[:remaining_words])
                        truncated_solutions.append(partial_solution + "...")
                    break
            
            reddit_solutions = truncated_solutions
            logger.info(f"Truncated reddit solutions to {sum(len(s.split()) for s in reddit_solutions)} words")
        
        state.solution_filtered_posts = reddit_filtered_posts
        state.reddit_solutions = reddit_solutions

        #get llm solutions for pain points 
        self._update_progress("Generate Solutions", "Generating AI solutions for pain points...")
        all_llm_solutions = [] 
        for summ_pain_point in summarized_pain_points:
            text = f"Pain point: {summ_pain_point.theme_name}\nDescription: {summ_pain_point.description}"
            llm_solutions = await self.llm_manager.generate_each_solutions(state.project_idea, 
                                                                      text)
            all_llm_solutions.append(llm_solutions)
        
        logger.info(f"Found {len(all_llm_solutions)} LLM solutions")
        logger.info(f"LLM solutions length : {sum(len(s) for s in all_llm_solutions)}")
        self._update_progress("Generate Solutions", f"Found {len(all_llm_solutions)} LLM solutions")
        state.llm_solutions = all_llm_solutions 

        #summarize llm solutions  
        logger.info(f"Summarizing LLM solutions...") 
        self._update_progress("Generate Solutions", "Summarizing LLM solutions...")
        llm_solutions_summary = await self.llm_manager.summarize_llm_solutions(state.project_idea, 
                                                                              all_llm_solutions,
                                                                              all_summarized_pain_points,
                                                                              reddit_solutions)
        state.summarized_llm_solutions = llm_solutions_summary
        self._update_progress("Generate Solutions", "LLM solutions summarized")
        return state

    async def _generate_report(self, state: ResearchState) -> ResearchState: 
        """Generate a report of the research using LLM""" 
        logger.info("Generating report...") 
        self._update_progress("Generate Report", "Generating report...")

        viz_files = self.report_manager.generate_visualizations()
        markdown_content = self.report_manager.generate_markdown(state)
        pdf_path = self.report_manager.generate_pdf_report(markdown_content, viz_files)

        self._update_progress("Generate Report", "Report generated")

        state.report_path = pdf_path
        return state

    async def run_research(self, project_id: str, project_idea: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete research workflow"""
        logger.info(f"Starting research for project: {project_id} - {project_idea}")

        #initialize state 
        initial_state = ResearchState(
            project_id=project_id,
            project_idea=project_idea,
            config=config
        )
        
        final_state = await self.workflow.ainvoke(initial_state)
        logger.info(f"Research complete!")

        # Save final state as pretty JSON
        base_path = os.path.join(self.projects_path, str(project_id))
        os.makedirs(base_path, exist_ok=True)
        
        # Convert Pydantic model to dict and save as JSON
        final_state_json_path = os.path.join(base_path, f"final_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(final_state_json_path, "w", encoding='utf-8') as f:
            json.dump(final_state, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Final state saved to: {final_state_json_path}")
        
        response = {
            "status": "success",
            "project_id": project_id,
            "final_state_path": final_state_json_path,
            "final_state": final_state,
            "summary": {
                "keywords_found": len(final_state["keywords"]),
                "subreddits_found": len(final_state["subreddits"]),
                "posts_found": len(final_state["reddit_posts"]),
                "filtered_posts": len(final_state["filtered_posts"]),
                "comments_found": len(final_state["reddit_comments"]),
                "pain_points_identified": len(final_state["pain_points"]),
            }
        }

        return response
    
    