keywords_extractor_prompt = """
You are an expert market researcher specializing in Reddit discourse analysis. Your task is to generate 5-8 strategically chosen keywords that will uncover the most relevant discussions about a specific project idea on Reddit.

These keywords should act as precision search terms to find posts and comments where people are genuinely discussing, experiencing, or seeking solutions related to the project's domain.

<instructions>
Generate between 5-8 keywords (NEVER MORE THAN 8) that capture multiple dimensions of the project space. Each keyword must be smartly selected and strategically chosen:

**Core Product/Service Keywords:**
- Direct terms for the main technology, product, or service
- Industry-specific terminology and jargon
- Technical specifications and features that matter to users

**Problem-Focused Keywords:**
- Pain points and frustrations users explicitly mention
- Common complaints and recurring issues
- "I wish there was..." or "Why doesn't..." type expressions
- Error messages, bugs, or failure scenarios people discuss

**Solution-Seeking Keywords:**
- Terms people use when asking for recommendations
- Comparison language ("X vs Y", "better than", "alternative to")
- Words indicating research behavior ("looking for", "anyone tried", "reviews")

**Competitor and Market Keywords:**
- Existing tools, platforms, or solutions in the space
- Brand names and product names users mention
- Category names and market segments

**User Journey Keywords:**
- Terms related to getting started, learning, or adoption challenges
- Advanced use cases and power user discussions
- Integration, setup, and configuration topics

**Emotional and Contextual Keywords:**
- Expressions of frustration, excitement, or urgency
- Situational contexts where the problem occurs
- Workflow disruptions or efficiency concerns

**Language Variations:**
- Include both technical and casual terminology
- Consider abbreviations, acronyms, and slang
- Account for different expertise levels (beginner vs expert language)
- Include typos or common misspellings of technical terms

CRITICAL: You must NEVER generate more than 8 keywords. If fewer high-quality keywords exist, provide only those (minimum 5, maximum 8).

All keywords must be smartly selected - avoid generic buzzwords, overly broad terms that would return irrelevant results, or keywords that stray from the core project focus. Each keyword should have a clear connection to either the problem the project solves or the solution it provides.

Balance specificity with discoverability - keywords should be narrow enough to find relevant content but broad enough to capture the full conversation landscape.
</instructions>

<response_format>
Return your response as a JSON object following this exact format:
{json_schema}
</response_format> 

Project Idea: {project_idea}
"""

filter_posts_prompt = """
You are a market research expert helping to decide which Reddit posts are actually useful for researching a specific project idea.

Your job is to score each Reddit post from 1 to 10 based on how relevant it is to the project idea. The more relevant and useful the post is for understanding the project area, the higher the score. The less relevant or off-topic, the lower the score.

<instructions>
When scoring posts, consider:
- Does the post talk about the same problem or topic as the project idea?
- Are people discussing pain points, frustrations, or needs related to the project?
- Does the post mention tools, solutions, or competitors in the same space?
- Are users asking questions or sharing experiences relevant to the project?
- Is the content detailed enough to give useful insights?

Scoring guide:
- 9-10: Highly relevant - directly discusses the project topic, problems, or solutions
- 7-8: Very relevant - related to the project area with useful insights
- 5-6: Somewhat relevant - touches on the topic but not deeply
- 3-4: Barely relevant - only loosely connected to the project
- 1-2: Not relevant - unrelated or off-topic content

Look at both the post title and content when making your decision.
</instructions>

<response_format>
Return your response as a JSON object following this exact format:
{json_schema}
</response_format>

project idea: {project_idea}
reddit post content: {post_content}
"""

pain_points_extractor_prompt = """
You are a helpful market research assistant. I need you to read through some Reddit content (a post and its comments) and find real problems or frustrations that people are talking about.
You need to tell me the real pain points people mention — but only if they are related to the project idea.

<instructions>
Specifically, look for:
- Things people complain about or find annoying
- Problems they wish were solved
- Things that seem to frustrate or annoy them
- Difficulties they have when doing something
- Stuff that they say doesn't work well or is broken
- Pain points that are related to the project idea

Look through both the original post content and all the comment responses to get a complete picture of what people are struggling with.
</instructions>

<important>
- Don't include random or unrelated issues. If the pain point isn't connected to the project idea, ignore it. 
- If there are no relevant problems mentioned in either the post or comments, just return an empty list.
- If the content is not related to the project idea, just return an empty list.
- Consider pain points mentioned in both the post and the comment discussions.
</important>

Keep your response simple and conversational - like you're telling a friend what you noticed people complaining about in the post and comment thread.

<response_format>
Return your response as a JSON object following this exact format:
{json_schema}
</response_format> 

project idea: {project_idea} 
Reddit post Text: {post_text}
Reddit post comments: {post_comments}
"""

pain_point_categorizer_prompt = """
You're helping us organize and understand user pain points for a specific project idea by putting them into the right category. I'll give you a user problem or frustration, and you will choose the best category that describes the type of issue.

<instructions>
Use the following categories:
**User Experience & Interface** - Interface is confusing, hard to navigate, poor design, not intuitive, or lacks accessibility features.
**Performance & Speed** - Slow loading times, laggy responses, poor optimization, or resource-intensive operations.
**Cost & Pricing** - Too expensive, unclear pricing models, hidden fees, poor value proposition, or billing issues.
**Feature Gaps & Functionality** - Missing essential features, limited capabilities, broken functionality, or inadequate feature depth.
**Integration & Compatibility** - Problems with third-party integrations, platform compatibility, API limitations, or ecosystem connectivity.
**Learning Curve & Documentation** - Steep learning curve, poor documentation, lack of tutorials, or insufficient onboarding.
**Technical Reliability** - System crashes, bugs, downtime, error handling, or inconsistent behavior.
**Data & Privacy Concerns** - Data security issues, privacy violations, compliance problems, or data portability concerns.
**Customer Support & Community** - Poor customer service, lack of support channels, unresponsive help, or weak community resources.
**Scalability & Growth Limitations** - Performance degradation with scale, resource limits, or inability to handle growing needs.
**Workflow & Productivity** - Inefficient processes, time-consuming tasks, repetitive manual work, or poor automation.
**Mobile & Cross-Platform Issues** - Mobile app problems, cross-device sync issues, or platform-specific limitations.
**Customization & Flexibility** - Lack of customization options, rigid workflows, or inability to adapt to specific use cases.
**Market & Competition** - Limited alternatives, vendor lock-in, competitive disadvantages, or industry-specific challenges.
**Other** - Anything that doesn't fit well into the above categories.
</instructions>

<important>
- Read the pain point carefully.
- Think about the main problem the user is facing.
- Choose the single best category from the list above.
- Consider both the technical and business aspects of the pain point.
</important>

<response_format>
Return your response as a JSON object following this exact format:
{json_schema}
</response_format>

project idea: {project_idea}
pain point: {pain_point}
"""

summarize_pain_points_prompt = """
You are helping us create a well-structured summaries of pain points related to a specific project idea.
I will give you a list of pain points. Your job is to analyze them and return a clear, markdown-formatted summaries.

<instructions>
What You Need to Do:
- **Filter for Relevance**: Only include pain points that are directly related to the project idea. Ignore anything that's not connected or relevant.
- **Group & Summarize Themes**: Combine similar or related pain points. Organize them under clear themes (e.g., Usability, Missing Features, etc.). Give each theme a heading and write a short summary of the issues under that theme.
1. **[Theme/Category]**: [Specific pain point]
- **Key Insights**: End the summary with a short section that summarizes the main takeaways and patterns from the analysis.
</instructions>

<important>
- ONLY include pain points that are directly related to the project idea. Exclude any pain points that are not relevant or connected to the specific project.
- Group similar or related pain points under a single theme to avoid repetition and redundancy.
- If there are fewer than 10 relevant pain points, return only the valid ones (do not create fake or additional pain points).
- Use clear, professional, and reader-friendly language throughout the report.
- Ensure each theme has a descriptive title that clearly represents the grouped pain points.
- Prioritize pain points based on severity, frequency, and relevance to the project idea.
</important>

<response_format>
The summarized pain points content should follow this exact markdown structure:

## Pain Points Summary

### [Theme Name - e.g., User Interface Challenges]
[Write a comprehensive summary paragraph that combines and describes all related pain points under this theme. Include specific details and examples from the pain points.]

### [Theme Name - e.g., Performance and Speed Issues]
[Write a comprehensive summary paragraph that combines and describes all related pain points under this theme. Include specific details and examples from the pain points.]

[Continue with additional themes as needed...]

## Key Insights

- [Identify and describe the most common patterns or recurring frustrations across multiple pain points]
- [Highlight the most critical issues that could significantly impact user experience or project success]
- [Summarize any emerging trends or unexpected findings from the pain point analysis]
- [Provide actionable insights that could guide product development or improvement priorities] 

Return your response as a JSON object following this exact format:
{json_schema}
</response_format>

project idea: {project_idea} 
pain points: {pain_points}

"""

solution_keywords_extractor_prompt = """
You are a market research expert helping to find useful keywords that people might use when talking about solutions to specific pain points on Reddit.

Your job is to generate a maximum of 8 clear, focused keywords that will help us find posts or comments where people are discussing solutions, workarounds, tools, or approaches to address the given pain points — not random or unrelated topics.

<instructions>
Generate between 3-8 keywords (NEVER MORE THAN 8) that capture multiple dimensions of the project space. Each keyword must be smartly selected and strategically chosen:

When thinking of keywords, make sure they:
- Are about solutions, fixes, or workarounds for the pain points mentioned.
- Include tools, software, or services that solve these problems.
- Mention alternative approaches or methods people use to address these issues.
- Refer to competitors or existing solutions people might already discuss.
- Include common words or phrases people use when asking for help or recommendations.
- Can include casual or slang terms, but still stay relevant to solving the pain points.
- Consider synonyms, alternative terms, and related expressions that users might employ when discussing solutions.
- Think about variations in terminology (technical vs. casual language, brand names vs. generic terms).
- Include both direct solution keywords and indirect ways people might reference fixes or improvements.
- Include keywords about success stories, recommendations, or positive experiences with solutions.
- Consider terms people use when sharing advice or suggesting solutions to others.

CRITICAL: You must NEVER generate more than 8 keywords. If fewer high-quality keywords exist, provide only those (minimum 3, maximum 8).

Don't add unrelated buzzwords, too-general terms, or things that drift away from solving the core pain points.
Make the keywords specific enough to find relevant solution-focused content, but not so narrow that you miss important discussions about fixes and alternatives.
Consider that users might express the same solutions using different vocabulary, so include synonymous terms and related phrases that capture the same intent.
Focus on both solution-oriented keywords and recommendation-focused keywords that reveal how people overcome these challenges.
</instructions>

<response_format>
Return your response as a JSON object following this exact format:
{json_schema}
</response_format> 

pain points: {pain_points}
"""

solution_filter_prompt = """
You are a market research expert helping to decide which Reddit posts about solutions are actually useful for researching a specific project idea and its related pain points.

Your job is to score each Reddit post from 1 to 10 based on how relevant it is to the project idea and whether it discusses solutions that could address the identified pain points. Additionally, if the post contains useful solutions, extract and summarize them in 50-100 words that relates to the pain points.

<instructions>
When scoring solution posts, consider:
- Does the post discuss solutions, tools, or approaches relevant to the project idea?
- Are people sharing fixes, workarounds, or recommendations that address the pain points?
- Does the post mention software, services, or methods that solve similar problems?
- Are users providing advice, tutorials, or success stories related to solving these issues?
- Is the content detailed enough to give useful insights about potential solutions?
- Does the solution discussion align with the pain points users are facing?

When extracting solutions, focus on:
- Specific tools, technologies, or approaches mentioned
- Step-by-step fixes or workarounds provided
- Recommended software, services, or methodologies
- Best practices or optimization techniques shared
- Alternative approaches or implementation strategies
- Success stories with actionable details

Provide the solution in exactly 50-100 words, focusing on the most relevant details from the Reddit post that address the pain points.

Scoring guide:
- 9-10: Highly relevant - directly discusses solutions for the project topic and addresses the pain points
- 7-8: Very relevant - related solutions with useful insights for addressing the pain points
- 5-6: Somewhat relevant - touches on solutions but not deeply connected to the pain points
- 3-4: Barely relevant - only loosely connected to project solutions or pain points
- 1-2: Not relevant - unrelated or off-topic solution content

Look at both the post title and content when making your decision, and consider how well the solutions align with the pain points provided.
</instructions>

<response_format>
Return your response as a JSON object following this exact format:
{json_schema}

If score is 6 or above, include extracted solutions in exactly 50-100 words that directly relate to the pain points. If score is below 6, solutions can be empty or null.
</response_format>

project idea: {project_idea}
pain points: {pain_points}
reddit post content: {post_content}
"""

generate_solutions_prompt = """
You are a product strategy expert helping to develop innovative solutions based on market research findings.

I will provide you with a single pain point from Reddit discussions related to a specific project idea. Your task is to analyze this pain point and generate a detailed, practical solution that directly addresses the identified problem.

<instructions>
What You Need to Do:
- **Analyze the Pain Point**: Thoroughly understand the specific problem users are facing and its impact
- **Generate Detailed Solution**: Provide a 50-100 word solution that directly addresses this pain point
- **Focus on Implementation**: Include specific technical approaches, features, and implementation strategies
- **Consider User Experience**: Address both the technical solution and how it improves the user experience
- **Be Actionable**: Ensure the solution is specific enough to guide development decisions and implementation
</instructions>

<important>
- The solution must directly address the specific pain point provided
- Write 50-100 words explaining the solution in detail
- Focus on practical, implementable approaches rather than vague suggestions
- Include specific technical features, methods, or approaches
- Consider both immediate implementation and long-term benefits
- Make the solution actionable and development-ready
- Address how this solution integrates with the overall project idea
</important>

<response_format>
Return your response as a JSON object following this exact format:
{json_schema}
</response_format> 

project idea: {project_idea}
pain point: {pain_point}
"""

summarize_llm_solutions_prompt = """
You are a product strategy expert helping to synthesize and organize solutions based on market research findings and pain point analysis.

I will provide you with:
1. A project idea
2. Categorized pain points with themes and descriptions
3. LLM-generated solutions for each pain point
4. Reddit-sourced solutions from real users

Your task is to analyze all these solutions and create a comprehensive, well-organized summary that groups solutions by the same themes as the pain points provided.

<instructions>
What You Need to Do:
- **Theme-Based Organization**: Group solutions under the exact same theme names provided in the pain points
- **Solution Integration**: Combine and synthesize both LLM-generated and Reddit-sourced solutions for each theme
- **Comprehensive Analysis**: For each theme, provide a detailed solution approach that addresses the pain point description
- **Practical Implementation**: Focus on actionable, implementable solutions with specific technical approaches
- **User-Centric Language**: Use the same type of language and terminology found in the pain points and user discussions
</instructions>

<important>
- Use the exact same theme names as provided in the pain points
- For each theme, create comprehensive solutions that directly address the pain point description
- Integrate insights from both LLM solutions and Reddit solutions
- Maintain consistency with user language and technical terminology
- Focus on practical, development-ready solutions
- Address how solutions integrate with the overall project idea
- Ensure solutions are specific enough to guide implementation decisions
</important>

<response_format>
Return your response as a JSON object following this exact format:
{json_schema}
</response_format>

project idea: {project_idea}
pain points: {pain_points}
llm solutions: {llm_solutions}
reddit solutions: {reddit_solutions}
"""

"""
**Solutions**:
1. **[Solution Name]**: [Detailed description of the solution, including how it addresses the pain point, key features, and implementation approach]

2. **[Solution Name]**: [Detailed description of the solution, including how it addresses the pain point, key features, and implementation approach]

3. **[Solution Name]**: [Detailed description of the solution, including how it addresses the pain point, key features, and implementation approach]

[Repeat for each pain point theme...]

## Strategic Recommendations

### Priority Features
- [List the top 3-5 features that should be prioritized based on pain point severity and user impact]

### Competitive Advantages
- [Identify unique solution approaches that could differentiate this project from competitors]

## Market Opportunity
- [Summarize the market opportunity based on the pain points and proposed solutions]
- [Estimate potential user impact and market size if these solutions were implemented]

Return your response as a JSON object following this exact format:
{json_schema}
</response_format>

Project Idea: {project_idea}
Summarized Pain Points: {summarized_pain_points}
"""

report_prompt_new = """
You are helping us create a well-structured report that analyzes and summarizes real user pain points related to a specific project idea.
I will give you a list of user pain points. Your job is to analyze them and return a clear, markdown-formatted report.

<instructions>
What You Need to Do:

1. **Filter for Relevance**: Only include pain points that are directly related to the project idea. Ignore anything that's not connected or relevant.

2. **Group & Summarize Themes**: Combine similar or related pain points. Organize them under clear themes (e.g., Usability, Missing Features, etc.). Give each theme a heading and write a short summary of the issues under that theme.

3. **Key Insights**: End the report with a short section that summarizes the main takeaways and patterns from the analysis.
</instructions>


<important>
- ONLY include pain points that are directly related to the project idea. Exclude any pain points that are not relevant or connected to the specific project.
- Use markdown format exclusively - no other formatting is allowed.
- Group similar or related pain points under a single theme to avoid repetition and redundancy.
- If there are fewer than 10 relevant pain points, return only the valid ones (do not create fake or additional pain points).
- Use clear, professional, and reader-friendly language throughout the report.
- Ensure each theme has a descriptive title that clearly represents the grouped pain points.
- Prioritize pain points based on severity, frequency, and relevance to the project idea.
</important>

<response_format>
The report content should follow this exact markdown structure:

## Pain Points Summary

### [Theme Name - e.g., User Interface Challenges]
[Write a comprehensive summary paragraph that combines and describes all related pain points under this theme. Include specific details and examples from the pain points.]

### [Theme Name - e.g., Performance and Speed Issues]
[Write a comprehensive summary paragraph that combines and describes all related pain points under this theme. Include specific details and examples from the pain points.]

[Continue with additional themes as needed...]

## Key Insights

- [Identify and describe the most common patterns or recurring frustrations across multiple pain points]
- [Highlight the most critical issues that could significantly impact user experience or project success]
- [Summarize any emerging trends or unexpected findings from the pain point analysis]
- [Provide actionable insights that could guide product development or improvement priorities] 

Return your response as a JSON object following this exact format:
{json_schema}
</response_format>

project idea: {project_idea} 
pain points: {pain_points}
"""
