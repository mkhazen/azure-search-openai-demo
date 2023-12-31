import os
from typing import Any, AsyncGenerator, Optional, Union

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI

from approaches.approach import Approach, ThoughtStep
from core.authentication import AuthenticationHelper
from core.messagebuilder import MessageBuilder

# Replace these with your own values, either in environment variables or directly here
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")


class RetrieveThenReadApproach(Approach):
    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

system_chat_template = (
    "You are an intelligent assistant helping analyze, translate and extracting information from legal documents from different countries such as Algeria and Morocco, The documents contain different languages Arabic, French and English text, graphs, tables and images. "
    + "Only answer in English, if the text is in Arabic or French translate it to English and extract the answer"
    +"Each image source has the file name in the top left corner of the image with coordinates (10,10) pixels and is in the format SourceFileName:<file_name> "
    + "Each text source starts in a new line and has the file name followed by colon and the actual information "
    + "Always include the source name from the image or text for each fact you use in the response in the format: [filename] "
    + "Answer the following question using only the data provided in the sources below. "
    + "For tabular information return it as an html table. Do not return markdown format. "
    +"Outputs should use exact contract language unless told specifically to summarize. Outputs in the form of tables may be useful for some prompts."
    +"Be precise in your answers, even extract the sentences as is from the document."
    +"It will be important to understand if the output is the exact same language or if it was summarize."
    +"Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below."
    +"If asking a clarifying question to the user would help, ask the question."
    +"For tabular information return it as an html table. Do not return markdown format."
    +"If the question is not in English, answer in the language used in the question."
    +"If there are multiple answers then either ask a clarifying question also if there are multiple answers rank all of these answers and provide all the answers ranked from highest confidence to lowest."
    +"Before you answer a question review the answer and ensure it is correct. Think step by step when you answer an answer to ensure it is correct."
    +"Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response." 
    +"Use square brackets to reference the source, e.g. [info1.txt]."
    +"Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf]."
    + "Each source has a name followed by colon and the actual data, quote the source name for each piece of data you use in the response. "'For example, if the question is "Who the buyer in the School House PPA" and one of the information sources says "info123: the buyer is Constellation NewEnergy", then answer with the exact answer from source and including it in quotation mark plus include the document [info123]" '
    +"'It's important to strictly follow the format where the name of the source is in square brackets at the end of the sentence, and only up to the prefix before the colon"
    +"'If there are multiple sources, cite each one in their own square brackets. For example, use '[info343][ref-76]' and not [info343,ref-76]"
    +"If you cannot answer using the sources below, say to provide clarifying question or provide an example."
    +"You can access to the following tools:"
    + "Use 'you' to refer to the individual asking the questions even if they ask with 'I'."
    + "Answer the following question using only the data provided in the sources below."
    + "For tabular information return it as an html table. Do not return markdown format."
    + "Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response."
    + "If you cannot answer using the sources below, say you don't know. Use below example to answer"
    + "The text and image source can be the same file name, don't use the image title when citing the image source, only use the file name as mentioned "
    + "If you cannot answer using the sources below, say you don't know. Return just the answer without any input texts "
)

# shots/sample conversation
question = """Could you compare the investment climates of Tunisia and Algeria based on the 2023 Investment Climate Statements, focusing on government incentives, market opportunities, and investment policies?"

Sources:
[0038 - Algeria - 2023 Investment Climate Statements.pdf]: The government passed a new hydrocarbon law in 2019, improving fiscal terms and contract flexibility in order to attract new international investors.
[0117 - Tunisia - 2023 Investment Climate Statements.pdf]: The GOT prioritizes attracting and retaining investment, particularly in the underdeveloped interior regions, and reducing unemployment by providing tax breaks, subsidizing social security fee contributions for new hires, and offering investment bonuses.
"""
answer = """ Algeria:
1.	Government Incentives: Offers incentives for investments in non-hydrocarbon sectors to diversify the economy.
2.	Market Opportunities: Rich in natural resources, particularly in the hydrocarbons sector.
3.	Investment Policies: Encourages foreign investments with a focus on local manufacturing and service sectors. [0038 - Algeria - 2023 Investment Climate Statements.pdf]
Tunisia:
1.	Government Incentives: Provides tax breaks and investment bonuses, especially in underdeveloped regions.
2.	Market Opportunities: Strategically located for trade, significant in agribusiness and aerospace.
3.	Investment Policies: While offering incentives, faces challenges in terms of political stability and bureaucracy.
Refer to '[0038 - Algeria - 2023 Investment Climate Statement.pdf]' for Algeria and Tunisia for detailed information [0117 - Tunisia - 2023 Investment Climate Statements.pdf] ."""
def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller

    async def run(
        self,
        messages: list[dict],
        stream: bool = False,  # Stream is not used in this approach
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> Union[dict[str, Any], AsyncGenerator[dict[str, Any], None]]:
        q = messages[-1]["content"]
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = overrides.get("semantic_ranker") and has_text

        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top", 3)
        filter = self.build_filter(overrides, auth_claims)
        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if has_vector:
            vectors.append(await self.compute_text_embedding(q))

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        query_text = q if has_text else None

        results = await self.search(top, query_text, filter, vectors, use_semantic_ranker, use_semantic_captions)

        user_content = [q]

        template = overrides.get("prompt_template") or self.system_chat_template
        model = self.chatgpt_model
        message_builder = MessageBuilder(template, model)

        # Process results
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

        # Append user message
        content = "\n".join(sources_content)
        user_content = q + "\n" + f"Sources:\n {content}"
        message_builder.insert_message("user", user_content)
        message_builder.insert_message("assistant", self.answer)
        message_builder.insert_message("user", self.question)

        chat_completion = (
            await self.openai_client.chat.completions.create(
                # Azure Open AI takes the deployment name as the model name
                model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
                messages=message_builder.messages,
                temperature=overrides.get("temperature") or 0.3,
                max_tokens=1024,
                n=1,
            )
        ).model_dump()

        data_points = {"text": sources_content}
        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Search Query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                    },
                ),
                ThoughtStep("Results", [result.serialize_for_results() for result in results]),
                ThoughtStep("Prompt", [str(message) for message in message_builder.messages]),
            ],
        }

        chat_completion["choices"][0]["context"] = extra_info
        chat_completion["choices"][0]["session_state"] = session_state
        return chat_completion
