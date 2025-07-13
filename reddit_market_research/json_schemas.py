from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class RedditKeywords(BaseModel): 
    keywords: List[str] = Field(description="List of keywords")

class SolutionKeywords(BaseModel):
    keywords: List[str] = Field(description="List of Solution keywords")

class PostScore(BaseModel):
    score: int = Field(description="Score of the post")

class SolutionPostScore(BaseModel):
    solution: str = Field(description="Solution of the project idea and pain point in 50-100 words")
    score: int = Field(description="Score of the solution post")

class PainPoints(BaseModel):
    pain_points: List[str] = Field(description="List of pain points")

class PainPointCategory(BaseModel):
    category: str = Field(description="Category of the pain point")

class PainPoint(BaseModel):
    """Data class for identified pain points"""
    id: str = Field(description="Unique identifier for the pain point")
    content: str = Field(description="Content of the pain point")
    category: str = Field(description="Category of the pain point")
    sources_post: str = Field(description="Post ID that contributed to the pain point")

class RedditPost(BaseModel):
    """Reddit post information"""
    id: str
    title: str
    content: str
    subreddit: str
    score: int
    num_comments: int
    created_utc: float
    url: str
    author: str
    flair: Optional[str] = None

class RedditComment(BaseModel):
    """Reddit comment information"""
    id: str
    post_id: str
    content: str
    score: int
    created_utc: float
    author: str
    parent_id: Optional[str] = None
    depth: int = 0
    upvotes: int = 0
    downvotes: int = 0

class KeyInsights(BaseModel):
    insight: str = Field(description="Insight about the pain points")

class EachSummarizedPainPoint(BaseModel): 
    theme_name: str = Field(description="Theme of the pain point") 
    description: str = Field(description="Description of the pain point") 

class SummarizedPainPoints(BaseModel):
    summarized_pain_points: List[EachSummarizedPainPoint] = Field(description="Summarized pain points") 
    key_insights: KeyInsights = Field(description="Key insights about the pain points")

class Solution(BaseModel):
    """Individual solution for a pain point theme"""
    solution_name: str = Field(description="Name of the solution")
    description: str = Field(description="Detailed description of the solution")

class ThemeSolutions(BaseModel):
    """Solutions for a specific pain point theme"""
    theme_name: str = Field(description="Name of the pain point theme")
    problem_summary: str = Field(description="Brief summary of the problems in this theme")
    solutions: List[Solution] = Field(description="List of solutions for this theme")

class StrategicRecommendations(BaseModel):
    """Strategic recommendations section"""
    priority_features: List[str] = Field(description="Top priority features")
    competitive_advantages: List[str] = Field(description="Unique competitive advantages")

class MarketOpportunity(BaseModel):
    """Market opportunity analysis"""
    opportunity_summary: str = Field(description="Summary of market opportunity")
    potential_impact: str = Field(description="Estimated potential user impact and market size")

class GeneratedSolutions(BaseModel):
    """Complete solutions response structure"""
    proposed_solutions: List[ThemeSolutions] = Field(description="Solutions organized by pain point themes")
    strategic_recommendations: StrategicRecommendations = Field(description="Strategic recommendations")
    market_opportunity: MarketOpportunity = Field(description="Market opportunity analysis")

class LLMSolution(BaseModel):
    solution: str = Field(description="Solution in detail")

class EachSummarizedLLMSolution(BaseModel): 
    theme_name: str = Field(description="Theme of the pain point")
    solution: str = Field(description="Solution in detail")

class SummarizedLLMSolutions(BaseModel): 
    summarized_llm_solutions: List[EachSummarizedLLMSolution] = Field(description="Solutions organized by pain point themes")

class ResearchState(BaseModel):
    """State management for the research workflow"""
    project_id: int = Field(description="The project ID")
    project_idea: str = Field(description="The project idea to research")
    config: Dict[str, Any] = Field(description="Configuration for the research workflow")
    keywords: List[str] = Field(default=[], description="List of keywords")
    subreddits: List[str] = Field(default=[], description="List of subreddits")
    reddit_posts: List[RedditPost] = Field(default=[], description="List of Reddit posts")
    filtered_posts: List[RedditPost] = Field(default=[], description="List of filtered Reddit posts based on the project idea")
    reddit_comments: List[RedditComment] = Field(default=[], description="List of Reddit comments")
    pain_points: List[PainPoint] = Field(default=[], description="List of pain points")
    summarized_pain_points: Optional[SummarizedPainPoints] = Field(default=None, description="List of summarized pain points")
    solution_keywords: List[str] = Field(default=[], description="List of solution keywords")
    solution_reddit_posts: List[RedditPost] = Field(default=[], description="List of Reddit posts for solutions")
    solution_filtered_posts: List[RedditPost] = Field(default=[], description="List of filtered Reddit posts for solutions")
    reddit_solutions: List[str] = Field(default=[], description="List of Reddit solutions")
    llm_solutions: List[str] = Field(default=[], description="List of LLM solutions")
    summarized_llm_solutions: Optional[SummarizedLLMSolutions] = Field(default=None, description="Summarized LLM solutions")
    report_path: str = Field(default="", description="Path to the report")