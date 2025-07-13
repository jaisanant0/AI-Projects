import os
from datetime import datetime

import praw
from prawcore.exceptions import ResponseException, RequestException

from pydantic import BaseModel, Field 
from typing import Optional, List
import asyncio
import logging
from collections import Counter, defaultdict
from json_schemas import RedditPost, RedditComment
from dotenv import load_dotenv
load_dotenv() 

# Configure logging with pretty terminal formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



class RedditAPIManager:
    """Manages Reddit API interactions with rate limiting""" 

    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )

        self.last_request_time = 0
        self.min_request_interval = 0.65 # 90 requests per minute 
    
    async def rate_limited_request(self, func, *args, **kwargs):
        """Rate limit requests to Reddit API""" 
        current_time = datetime.now().timestamp()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        try:
            result = func(*args, **kwargs) 
            self.last_request_time = datetime.now().timestamp()
            return result
        
        except (ResponseException, RequestException) as e:
            logger.error(f"Reddit API error: {e}")
            if "rate limit" in str(e).lower():
                await asyncio.sleep(60)
                return await self.rate_limited_request(func, *args, **kwargs)
            raise e

    async def get_subreddits(self, keywords: List[str]) -> List[str]:
        """Get subreddits from keywords"""
        all_subreddits = []
        for keyword in keywords:
            subreddit = await self.rate_limited_request(self.reddit.subreddit('all').search,
                                                        query=keyword,
                                                        sort = "relevance",
                                                        time_filter = "all",
                                                        limit = 100)
            counter = Counter()
            for post in subreddit: 
                counter[post.subreddit.display_name] += 1 
            subreddits = [sub for sub, count in counter.most_common(2)]
            all_subreddits.extend(subreddits)
        return list(set(all_subreddits))
    
    async def get_post_comments(self, post_id: str, limit: int = 50, min_comment_score: int = 2) -> List[RedditComment]:
        """Get comments for a specific post"""
        comments = []

        try:
            submission = await self.rate_limited_request(
                self.reddit.submission,
                post_id
            )

            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                if comment.score >= min_comment_score and len(comments) < limit and hasattr(comment, 'body') and comment.body not in ("[deleted]", "[removed]") and comment.author is not None:
                    reddit_comment = RedditComment(
                        id=comment.id,
                        post_id=post_id,
                        content=comment.body,
                        score=comment.score,
                        created_utc=comment.created_utc,
                        author=str(comment.author),
                        parent_id=comment.parent_id,
                        depth=comment.depth if hasattr(comment, 'depth') else 0,
                        upvotes=comment.ups,
                        downvotes=comment.downs
                    )
                    comments.append(reddit_comment)

            return comments
        
        except Exception as e:
            logger.error(f"Error getting comments for post {post_id}: {e}")
            return []
                
    async def search_subreddits(self, subreddits: List[str], query: str, limit: int = 10, 
                                 min_post_score: int = 10,
                                 num_comments: int = 5,
                                 strategies = [
                                     ("relevance", "all"),
                                     ("top", "month"),
                                     ("new", None)
                                     ]) -> List[RedditPost]:
        """Search Reddit posts with rate limiting"""
        all_posts = []

        try:
            for sub in subreddits:
                #logger.info(f"Searching {sub} with {query}")
                for sort, time_filter in strategies:
                    subreddit = self.reddit.subreddit(sub) 
                    if time_filter is None:
                        posts = await self.rate_limited_request( 
                            subreddit.search,
                            query=query,
                            limit=limit,
                            sort=sort
                        )
                    else:
                        posts = await self.rate_limited_request( 
                            subreddit.search,
                            query=query,
                            limit=limit,
                            sort=sort,
                            time_filter=time_filter 
                        )
                    for post in posts:
                        if post.score < min_post_score or post.num_comments < num_comments:
                            continue

                        post_data = RedditPost(
                            id=post.id,
                            title=post.title,
                            content=post.selftext,
                            subreddit=post.subreddit.display_name,
                            score=post.score,
                            num_comments=post.num_comments,
                            created_utc=post.created_utc,
                            url=post.url,
                            author=str(post.author),
                            flair=post.link_flair_text
                        )
                        all_posts.append(post_data)

            return all_posts
        
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
            return []
            
if __name__ == "__main__":
    reddit_manager = RedditAPIManager()
    subreddits = asyncio.run(reddit_manager.get_subreddits(["ai agent error"]))
    all_posts = asyncio.run(reddit_manager._search_subreddits(subreddits, "ai agent error")) 
    for post in all_posts:
        print(f"Title: {post.title}")
        print(f"Subreddit: {post.subreddit}")
        print(f"Score: {post.score}")
        print(f"Number of comments: {post.num_comments}")
        print(f"Author: {post.author}")
        print(f"Flair: {post.flair}")
        print(f"--------------------------------")