from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from approaches.approach import AskApproach
from core.messagebuilder import MessageBuilder
from text import nonewlines


class RetrieveThenReadApproach(AskApproach):
    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_chat_template = (
        "You are an Assistant thar helps the company employees analyze, compare, and extract information from wholesale, retail and standard form contracts and associated notices." 
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
    )

    # shots/sample conversation
    question = """
    'What is standard language for Force Majeure in a Wholesale PPA?'

Sources:
PJM Standard Form Solar PPA (Project-Specific RECs) 2023: “Force Majeure” means any event that wholly or partly prevents or delays the performance by the Party affected of any obligation arising hereunder, but only if and to the extent: (a) such event or condition is not reasonably foreseeable and is not within the reasonable control of the Party affected; (b) that despite the exercise of reasonable diligence, cannot be or be caused to be prevented or avoided by such Party; and (c) such event is not the direct or indirect result of the affected Party’s negligence or the failure of such Party to perform any of its obligations under this Agreement.  Events and conditions that may constitute Force Majeure include, to the extent satisfying the foregoing requirements, events and conditions that are within or similar to one or more of the following categories: condemnation; expropriation; invasion; plague; drought; landslide; hurricane or tropical storm; tornado; lightning, ice storm, dust storm, tsunami; flood; earthquake; fire; explosion; epidemic; pandemic; quarantine; war (declared or undeclared), terrorism or other armed conflict; strikes and other labor disputes if such strike or other labor dispute is not specifically directed at the affected Party; riot or similar civil disturbance or commotion; other acts of God; acts of the public enemy; blockade; insurrection, riot or revolution; sabotage or vandalism; embargoes; and failures or delays of the Transmission System Owner directly impacting the Facility or the Interconnection Facilities but only to the extent such failures or delays were due to circumstances would constitute a “Force Majeure” impacting the Transmission System Owner.  The term “Force Majeure” shall not include (A) a Party’s ability to enter into a contract for the hedge of energy and/or sale or purchase of the Products at a more favorable price or under more favorable conditions or other economic reasons, (B) delays or nonperformance by suppliers, vendors, or other third parties with whom a Party has contracted (including interconnection or permitting delays) except to the extent that such delays or nonperformance were due to circumstances that would constitute Force Majeure, (C) serial defects of the Facility’s equipment, (D) any other economic hardship or changes in market conditions affecting the economics of either Party, (E) any delay in providing, or cancellation of, any approvals or permits by the issuing governmental authority, except to the extent such delay or cancellation was due to circumstances that would constitute Force Majeure, (F) weather conditions (including severe and extreme weather, other than the weather events expressly provided above), (G) failure or breakdown of mechanical equipment except to the extent that such delays or nonperformance were due to circumstances that would constitute Force Majeure, (H) variability in solar irradiance, including periods of low solar irradiance resulting from cloud cover, pollution, dust, smoke, weather conditions, and other causes, in the area in which the Facility is located, or (I) impacts of the COVID-19 pandemic or any mutations thereof except to the extent that, notwithstanding clause (a) above, the affected Party was not aware of, and would not reasonably be expected to have been aware of, such impacts based on information available to the affected Party as of the Effective Date.
"""
    answer = "“Force Majeure” means any event that wholly or partly prevents or delays the performance by the Party affected of any obligation arising hereunder, but only if and to the extent: (a) such event or condition is not reasonably foreseeable and is not within the reasonable control of the Party affected; (b) that despite the exercise of reasonable diligence, cannot be or be caused to be prevented or avoided by such Party; and (c) such event is not the direct or indirect result of the affected Party’s negligence or the failure of such Party to perform any of its obligations under this Agreement.  Events and conditions that may constitute Force Majeure include, to the extent satisfying the foregoing requirements, events and conditions that are within or similar to one or more of the following categories: condemnation; expropriation; invasion; plague; drought; landslide; hurricane or tropical storm; tornado; lightning, ice storm, dust storm, tsunami; flood; earthquake; fire; explosion; epidemic; pandemic; quarantine; war (declared or undeclared), terrorism or other armed conflict; strikes and other labor disputes if such strike or other labor dispute is not specifically directed at the affected Party; riot or similar civil disturbance or commotion; other acts of God; acts of the public enemy; blockade; insurrection, riot or revolution; sabotage or vandalism; embargoes; and failures or delays of the Transmission System Owner directly impacting the Facility or the Interconnection Facilities but only to the extent such failures or delays were due to circumstances would constitute a “Force Majeure” impacting the Transmission System Owner.  The term “Force Majeure” shall not include (A) a Party’s ability to enter into a contract for the hedge of energy and/or sale or purchase of the Products at a more favorable price or under more favorable conditions or other economic reasons, (B) delays or nonperformance by suppliers, vendors, or other third parties with whom a Party has contracted (including interconnection or permitting delays) except to the extent that such delays or nonperformance were due to circumstances that would constitute Force Majeure, (C) serial defects of the Facility’s equipment, (D) any other economic hardship or changes in market conditions affecting the economics of either Party, (E) any delay in providing, or cancellation of, any approvals or permits by the issuing governmental authority, except to the extent such delay or cancellation was due to circumstances that would constitute Force Majeure, (F) weather conditions (including severe and extreme weather, other than the weather events expressly provided above), (G) failure or breakdown of mechanical equipment except to the extent that such delays or nonperformance were due to circumstances that would constitute Force Majeure, (H) variability in solar irradiance, including periods of low solar irradiance resulting from cloud cover, pollution, dust, smoke, weather conditions, and other causes, in the area in which the Facility is located, or (I) impacts of the COVID-19 pandemic or any mutations thereof except to the extent that, notwithstanding clause (a) above, the affected Party was not aware of, and would not reasonably be expected to have been aware of, such impacts based on information available to the affected Party as of the Effective Date."

    def __init__(
        self,
        search_client: SearchClient,
        openai_host: str,
        chatgpt_deployment: str,
        chatgpt_model: str,
        embedding_deployment: str,
        embedding_model: str,
        sourcepage_field: str,
        content_field: str,
    ):
        self.search_client = search_client
        self.openai_host = openai_host
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    async def run(self, q: str, overrides: dict[str, Any], auth_claims: dict[str, Any]) -> dict[str, Any]:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            embedding_args = {"deployment_id": self.embedding_deployment} if self.openai_host == "azure" else {}
            embedding = await openai.Embedding.acreate(**embedding_args, model=self.embedding_model, input=q)
            query_vector = embedding["data"][0]["embedding"]
        else:
            query_vector = None

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        query_text = q if has_text else ""

        # Use semantic ranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(
                query_text,
                filter=filter,
                query_type=QueryType.SEMANTIC,
                query_language="en-us",
                query_speller="lexicon",
                semantic_configuration_name="default",
                top=top,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                vector=query_vector,
                top_k=50 if query_vector else None,
                vector_fields="embedding" if query_vector else None,
            )
        else:
            r = await self.search_client.search(
                query_text,
                filter=filter,
                top=top,
                vector=query_vector,
                top_k=50 if query_vector else None,
                vector_fields="embedding" if query_vector else None,
            )
        if use_semantic_captions:
            results = [
                doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc["@search.captions"]]))
                async for doc in r
            ]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
        content = "\n".join(results)

        message_builder = MessageBuilder(
            overrides.get("prompt_template") or self.system_chat_template, self.chatgpt_model
        )

        # add user question
        user_content = q + "\n" + f"Sources:\n {content}"
        message_builder.append_message("user", user_content)

        # Add shots/samples. This helps model to mimic response and make sure they match rules laid out in system message.
        message_builder.append_message("assistant", self.answer)
        message_builder.append_message("user", self.question)

        messages = message_builder.messages
        chatgpt_args = {"deployment_id": self.chatgpt_deployment} if self.openai_host == "azure" else {}
        chat_completion = await openai.ChatCompletion.acreate(
            **chatgpt_args,
            model=self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature") or 0.3,
            max_tokens=1024,
            n=1,
        )

        return {
            "data_points": results,
            "answer": chat_completion.choices[0].message.content,
            "thoughts": f"Question:<br>{query_text}<br><br>Prompt:<br>"
            + "\n\n".join([str(message) for message in messages]),
        }
