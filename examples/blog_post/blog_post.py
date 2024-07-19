from dotenv import load_dotenv

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate

from motleycrew import MotleyCrew, Task
from motleycrew.agents.langchain.legacy_react import LegacyReActMotleyAgent
from motleycrew.tools.llm_tool import LLMTool
from .blog_post_input import text

load_dotenv()

# TODO: switch example to using URL instead of fixed text?
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import TokenTextSplitter
# def urls_to_messages(urls: Union[str, Sequence[str]]) -> List[HumanMessage]:
#     if isinstance(urls, str):
#         urls = [urls]
#     loader = UnstructuredURLLoader(urls=urls)
#     data = loader.load()
#     text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.split_documents(data)
#     return [HumanMessage(content=d.page_content) for d in texts]


max_words = 500
min_words = 450

editor_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are an experienced online blog post editor with 10 years of experience."
        ),
        HumanMessage(
            content="""Review the blog post draft below (delimited by triple backticks) 
        and provide a critique and use specific examples from the text on what 
    should be done to improve the draft, with data professionals as the intended audience. 
    Also, suggest a catchy title for the story.
       ```{input}```
    """
        ),
    ]
)

illustrator_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a professional illustrator with 10 years of experience."),
        HumanMessage(
            content="You are given the following draft story, delimited by triple back quotes: ```{second_draft}```"
        ),
        HumanMessage(
            content="""Your task is to specify the illustrations that would fit this story. 
    Make sure the illustrations are varied in style, eye-catching, and some of them humorous.
    Describe each illustration in a way suitable for entering in a Midjourney prompt.  
    Each description should be detailed and verbose. Don't explain the purpose of the illustrations, 
    just describe in great 
    detail what each illustration should show, in a way suitable for a generative image prompt.
    There should be at most 5 and at least 3 illustrations.
    Return the illustration descriptions as a list in the format 
    ["...", "...", ..., "..."]
    """
        ),
    ]
)

seo_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""Act as an SEO expert with 10 years of experience but ensure to 
            explain any SEO jargon for clarity when using it."""
        ),
        HumanMessage(
            content="""Review the blog post below (delimited by triple back quotes) and provide specific 
examples from the text where to optimize its SEO content. 
Recommend SEO-friendly titles and subtitles that could be used.
```{second_draft}```
"""
        ),
    ]
)

editor = LLMTool(
    name="editor",
    description="An editor providing constructive suggestions to improve the blog post submitted to it",
    prompt=editor_prompt,
)

# TODO: Turn it into an agent that calls the DALL-E tool
# and returns a dict {image_description: image_url}
illustrator = LLMTool(
    name="illustrator",
    description="An illustrator providing detailed descriptions of illustrations for a story",
    prompt=illustrator_prompt,
)

seo_expert = LLMTool(
    name="seo_expert",
    description="An SEO expert providing SEO optimization suggestions",
    prompt=seo_prompt,
)


writer = LegacyReActMotleyAgent(
    prompt="You are a professional freelance copywriter with 10 years of experience.",
    tools=[editor, illustrator, seo_expert],
)

# Create tasks for your agents
crew = MotleyCrew()
task1 = Task(
    crew=crew,
    name="Write a blog post from the provided information",
    description=f"""Write a blog post of at most {max_words} words and at least {min_words}
            words based on the information provided. Keep the tone suitable for an audience of
            data professionals, avoid superlatives and an overly excitable tone.
            Don't discuss installation or testing.
            The summary will be provided in one or multiple chunks, followed by <END>.
            
            Proceed as follows: first, write a draft blog post as described above.
            Then, submit it in turn to the editor, illustrator, and SEO expert for feedback.
            In the case of the illustrator, insert the illustration descriptions it provides in 
            square brackets into the appropriate places in the draft.
            In each case, revise the draft as per the response of the expert and submit it to the next expert.
            
            After you have implemented each expert's recommendations, return the final draft in markdown format.
            
            Return the blog post in markdown format. 
            Information begins: {text} <END>""",
    agent=writer,
)

crew.run(verbose=2)
