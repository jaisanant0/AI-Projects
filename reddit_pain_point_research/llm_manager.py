from openai import OpenAI
import os
import asyncio
from typing import List, Dict, Any
import json
from prompts import (
    keywords_extractor_prompt,
    pain_points_extractor_prompt,
    pain_point_categorizer_prompt,
    summarize_pain_points_prompt,
    generate_solutions_prompt,
    filter_posts_prompt,
    solution_keywords_extractor_prompt,
    solution_filter_prompt,
    summarize_llm_solutions_prompt
)

from json_schemas import (
    RedditKeywords,
    PainPoints,
    PainPoint,
    PainPointCategory,
    SummarizedPainPoints,
    GeneratedSolutions,
    RedditPost,
    RedditComment,
    PostScore,
    SolutionKeywords,
    SolutionPostScore,
    LLMSolution,
    SummarizedLLMSolutions,
    ResearchState
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class LLMManager: 
    def __init__(self, model_name: str, endpoint: str):
        self.model_name = model_name
        self.endpoint = endpoint
        self.llm = OpenAI(api_key="123", base_url=self.endpoint) 

    async def generate_keywords(self, project_idea: str, is_thinking: bool = False) -> List[str]:
        """Generate keywords for project idea"""

        prompt = keywords_extractor_prompt.format(
            json_schema=RedditKeywords.model_json_schema(),
            project_idea=project_idea
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        if is_thinking:
            model_kwargs = {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 512
            }

        else:
            model_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 512
            }

        response = self.llm.chat.completions.create(
            model=self.model_name, 
            messages=messages,
            **model_kwargs,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": is_thinking},
                "guided_json": RedditKeywords.model_json_schema(),
                "top_k": 20,
                "min_p": 0,
            }
        )

        keywords = json.loads(response.choices[0].message.content)
        return keywords

    async def filter_posts(self, project_idea: str, posts: List[RedditPost], filter_threshold: int = 7, is_thinking: bool = False) -> List[RedditPost]: 
        """Filter posts using LLM"""

        filtered_posts = []
        for post in posts: 
            prompt = filter_posts_prompt.format(
                json_schema=PostScore.model_json_schema(),
                project_idea=project_idea,
                post_content=post.content[:500]
            )

            messages = [
                {"role": "user", "content": prompt}
            ]

            if is_thinking:
                model_kwargs = {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "max_tokens": 128
                }

            else:
                model_kwargs = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_tokens": 128
                }
            
            response = self.llm.chat.completions.create(
                model=self.model_name, 
                messages=messages,
                **model_kwargs,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": is_thinking},
                    "guided_json": PostScore.model_json_schema(),
                    "top_k": 20,
                    "min_p": 0,
                }
            )
            score = json.loads(response.choices[0].message.content)
            post_score = score["score"]
            if post_score >= filter_threshold:
                filtered_posts.append(post)

        return filtered_posts
    
    async def filter_solution_posts(self, project_idea: str, posts: List[RedditPost], pain_points: str, is_thinking: bool = False, filter_threshold: int = 7) -> List[RedditPost]:
        """Filter solution posts using LLM"""
        filtered_posts = []
        solutions =[]
        for i, post in enumerate(posts): 
            logger.info(f"({i+1}/{len(posts)}) Filtering solution posts for post: {post.title}")
            prompt = solution_filter_prompt.format(
                json_schema=SolutionPostScore.model_json_schema(),
                project_idea=project_idea,
                pain_points=pain_points,
                post_content=post.content[:500]
            )

            messages = [
                {"role": "user", "content": prompt}
            ]

            if is_thinking:
                model_kwargs = {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "max_tokens": 4096
                }

            else:
                model_kwargs = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_tokens": 4096
                }

            response = self.llm.chat.completions.create(
                model=self.model_name, 
                messages=messages,
                **model_kwargs,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": is_thinking},
                    "guided_json": SolutionPostScore.model_json_schema(),
                    "top_k": 20,
                    "min_p": 0,
                }
            )
            
            score = json.loads(response.choices[0].message.content)
            post_score = score["score"]
            post_solution = score["solution"]
            if post_score >= filter_threshold:
                filtered_posts.append(post)
                solutions.append(post_solution)
        return filtered_posts, solutions
            
    async def extract_pain_points(self, project_idea: str, post: RedditPost, post_comments: List[RedditComment], is_thinking: bool = False) -> List[str]: 
        """Extract pain points from content"""
        
        comments_text = "" 
        for i, comment in enumerate(post_comments): 
            comments_text += f"Comment {i+1}:\n{comment.content}\n"

        prompt = pain_points_extractor_prompt.format(
            json_schema=PainPoints.model_json_schema(),
            project_idea=project_idea,
            post_text=post.content,
            post_comments=comments_text
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        if is_thinking:
            model_kwargs = {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 8000
            }

        else:
            model_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 8000
            }

        response = self.llm.chat.completions.create(
            model=self.model_name, 
            messages=messages,
            **model_kwargs,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": is_thinking},
                "guided_json": PainPoints.model_json_schema(),
                "top_k": 20,
                "min_p": 0,
            }
        )

        pain_points = json.loads(response.choices[0].message.content)
        return pain_points

    async def categorize_pain_point(self, project_idea: str, pain_point: str, is_thinking: bool = False) -> str:
        """Categorize pain point"""

        prompt = pain_point_categorizer_prompt.format(
            json_schema=PainPointCategory.model_json_schema(),
            project_idea=project_idea,
            pain_point=pain_point
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        if is_thinking:
            model_kwargs = {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 128
            }

        else:
            model_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 128
            }

        response = self.llm.chat.completions.create(
            model=self.model_name, 
            messages=messages,
            **model_kwargs,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": is_thinking},
                "guided_json": PainPointCategory.model_json_schema(),
                "top_k": 20,
                "min_p": 0,
            }
        )

        category = json.loads(response.choices[0].message.content)
        return category

    async def summarize_pain_points(self, project_idea: str, pain_points: str, is_thinking: bool = False) -> str:
        """Summarize pain points"""

        prompt = summarize_pain_points_prompt.format(
            json_schema=SummarizedPainPoints.model_json_schema(),
            project_idea=project_idea,
            pain_points=pain_points
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        if is_thinking:
            model_kwargs = {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 16000
            }

        else:
            model_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 16000
            }
        
        response = self.llm.chat.completions.create(
            model=self.model_name, 
            messages=messages,
            **model_kwargs,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": is_thinking},
                "guided_json": SummarizedPainPoints.model_json_schema(),
                "top_k": 20,
                "min_p": 0,
            }
        )

        summarized_pain_points = json.loads(response.choices[0].message.content)
        return summarized_pain_points

    async def generate_solutions_keywords(self, project_idea: str, pain_points: SummarizedPainPoints, is_thinking: bool = False) -> str: 
        """Generate solutions for pain points""" 
        summarized_pain_points = pain_points.summarized_pain_points
        all_summarized_pain_points = ""
        for summ_pain_point in summarized_pain_points:
            all_summarized_pain_points += f"{summ_pain_point.theme_name}: {summ_pain_point.description}\n"

        keyord_prompt = solution_keywords_extractor_prompt.format(
            json_schema=SolutionKeywords.model_json_schema(),
            pain_points=all_summarized_pain_points
        )

        messages = [
            {"role": "user", "content": keyord_prompt}
        ]

        if is_thinking:
            model_kwargs = {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 16000
            }

        else:
            model_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 16000
            }

        response = self.llm.chat.completions.create(
            model=self.model_name, 
            messages=messages,
            **model_kwargs,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": is_thinking},
                "guided_json": SolutionKeywords.model_json_schema(),
                "top_k": 20,
                "min_p": 0,
            }
        )

        solution_keywords = json.loads(response.choices[0].message.content)
        return solution_keywords

        '''prompt = generate_solutions_prompt.format(
            json_schema=SolutionKeywords.model_json_schema(),
            project_idea=project_idea,
            summarized_pain_points=all_summarized_pain_points
        )
        
        messages = [
            {"role": "user", "content": prompt}
        ]

        

        response = self.llm.chat.completions.create(
            model=self.model_name, 
            messages=messages,
            **model_kwargs,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": is_thinking},
                "guided_json": GeneratedSolutions.model_json_schema(),
                "top_k": 20,
                "min_p": 0,
            }
        )

        generated_solutions = json.loads(response.choices[0].message.content)
        return generated_solutions'''

    async def generate_each_solutions(self, project_idea: str, pain_point: str, is_thinking: bool = False) -> str:
        """Generate solutions for pain points"""

        prompt = generate_solutions_prompt.format(
            json_schema=LLMSolution.model_json_schema(),
            project_idea=project_idea,
            pain_point=pain_point
        )

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        if is_thinking:
            model_kwargs = {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 2048
            }

        else:
            model_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 2048
            }
        
        response = self.llm.chat.completions.create(
            model=self.model_name, 
            messages=messages,
            **model_kwargs,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": is_thinking},
                "guided_json": LLMSolution.model_json_schema(), 
                "top_k": 20,
                "min_p": 0,
            }
        )

        llm_solution = json.loads(response.choices[0].message.content)
        return llm_solution["solution"]
    
    async def summarize_llm_solutions(self, project_idea: str, llm_solutions: List[str], 
    pain_points: str, reddit_solutions: List[str], is_thinking: bool = False) -> str: 
        """Summarize LLM solutions""" 

        all_llm_solutions = ""
        for i,llm_solution in enumerate(llm_solutions):
            all_llm_solutions += f"Solution {i+1}:\n{llm_solution}\n"

        all_reddit_solutions = ""
        for i,reddit_solution in enumerate(reddit_solutions):
            all_reddit_solutions += f"Solution {i+1}:\n{reddit_solution}\n"

        prompt = summarize_llm_solutions_prompt.format(
            json_schema=SummarizedLLMSolutions.model_json_schema(),
            project_idea=project_idea,
            pain_points=pain_points,
            llm_solutions=all_llm_solutions,
            reddit_solutions=all_reddit_solutions
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        if is_thinking:
            model_kwargs = {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 16000
            }

        else:
            model_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 16000
            }

        response = self.llm.chat.completions.create(
            model=self.model_name, 
            messages=messages,
            **model_kwargs,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": is_thinking},
                "guided_json": SummarizedLLMSolutions.model_json_schema(),
                "top_k": 20,
                "min_p": 0,
            }
        )

        summarized_llm_solutions = json.loads(response.choices[0].message.content)
        return summarized_llm_solutions
    
       

if __name__ == "__main__": 
    llm_manager = LLMManager(
        model_name=os.getenv("MODEL_NAME"),
        endpoint=os.getenv("ENDPOINT")
    )

    keywords = asyncio.run(llm_manager.generate_keywords(
        project_idea="I want to build a startup that makes a product that helps people learn to code.", is_thinking=True
    ))