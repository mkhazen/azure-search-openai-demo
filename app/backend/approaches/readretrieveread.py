from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.callbacks.manager import CallbackManager, Callbacks
from langchain.chains import LLMChain
from langchain.llms.openai import AzureOpenAI, OpenAI

from approaches.approach import AskApproach
from langchainadapters import HtmlCallbackHandler
from lookuptool import CsvLookupTool
from text import nonewlines


class ReadRetrieveReadApproach(AskApproach):
    """
    Attempt to answer questions by iteratively evaluating the question to see what information is missing, and once all information
    is present then formulate an answer. Each iteration consists of two parts:
     1. use GPT to see if we need more information
     2. if more data is needed, use the requested "tool" to retrieve it.
    The last call to GPT answers the actual question.
    This is inspired by the MKRL paper[1] and applied here using the implementation in Langchain.

    [1] E. Karpas, et al. arXiv:2205.00445
    """

    template_prefix = (
        "You are an Assistant thar helps the company employees analyze, compare, and extract information from wholesale, retail and standard form contracts and associated notices." 
        "Outputs should use exact contract language unless told specifically to summarize. Outputs in the form of tables may be useful for some prompts."
        "Be precise in your answers, even extract the sentences as is from the document."
        "It will be important to understand if the output is the exact same language or if it was summarize."
        "Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below."
        "If asking a clarifying question to the user would help, ask the question."
        "For tabular information return it as an html table. Do not return markdown format."
        "If the question is not in English, answer in the language used in the question."
        "If there are multiple answers then either ask a clarifying question also if there are multiple answers rank all of these answers and provide all the answers ranked from highest confidence to lowest."
        "Before you answer a question review the answer and ensure it is correct. Think step by step when you answer an answer to ensure it is correct."
        "Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response." 
        "Use square brackets to reference the source, e.g. [info1.txt]."
        "Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf]."
        "Each source has a name followed by colon and the actual data, quote the source name for each piece of data you use in the response. "
        "For example, if the question is 'Who the buyer in the School House PPA' and one of the information sources says 'info123: the buyer is Constellation NewEnergy', then answer with the exact answer from source and including it in quotation mark plus include the document [info123]" 
        "It is important to strictly follow the format where the name of the source is in square brackets at the end of the sentence, and only up to the prefix before the colon"
        "If there are multiple sources, cite each one in their own square brackets. For example, use '[info343][ref-76]' and not [info343,ref-76]"
        "If you cannot answer using the sources below, say to provide clarifying question or provide an example."
        "You can access to the following tools:"
    )

    template_suffix = """
Begin!

Question: {input}

Thought: {agent_scratchpad}"""

    CognitiveSearchToolDescription = "useful for searching the Contract and notices etc."

    def __init__(
        self,
        search_client: SearchClient,
        openai_host: str,
        openai_deployment: str,
        openai_model: str,
        embedding_deployment: str,
        embedding_model: str,
        sourcepage_field: str,
        content_field: str,
    ):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.openai_model = openai_model
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.openai_host = openai_host

    async def retrieve(self, query_text: str, overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            embedding_args = {"deployment_id": self.embedding_deployment} if self.openai_host == "azure" else {}
            embedding = await openai.Embedding.acreate(**embedding_args, model=self.embedding_model, input=query_text)
            query_vector = embedding["data"][0]["embedding"]
        else:
            query_vector = None

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            query_text = ""

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
                doc[self.sourcepage_field] + ":" + nonewlines(" -.- ".join([c.text for c in doc["@search.captions"]]))
                async for doc in r
            ]
        else:
            results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field][:250]) async for doc in r]
        content = "\n".join(results)
        return results, content

    async def run(self, q: str, overrides: dict[str, Any]) -> dict[str, Any]:
        retrieve_results = None

        async def retrieve_and_store(q: str) -> Any:
            nonlocal retrieve_results
            retrieve_results, content = await self.retrieve(q, overrides)
            return content

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])

        acs_tool = Tool(
            name="CognitiveSearch",
            func=lambda _: "Not implemented",
            coroutine=retrieve_and_store,
            description=self.CognitiveSearchToolDescription,
            callbacks=cb_manager,
        )
        employee_tool = EmployeeInfoTool("Employee1", callbacks=cb_manager)
        tools = [acs_tool, employee_tool]

        prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=overrides.get("prompt_template_prefix") or self.template_prefix,
            suffix=overrides.get("prompt_template_suffix") or self.template_suffix,
            input_variables=["input", "agent_scratchpad"],
        )
        if self.openai_type == "azure":
            llm = AzureOpenAI(
                deployment_name=self.openai_deployment,
                temperature=overrides.get("temperature", 0.3),
                openai_api_key=openai.api_key,
            )
        else:
            llm = OpenAI(
                model_name=self.openai_model,
                temperature=overrides.get("temperature", 0.3),
                openai_api_key=openai.api_key,
            )

        chain = LLMChain(llm=llm, prompt=prompt)
        agent_exec = AgentExecutor.from_agent_and_tools(
            agent=ZeroShotAgent(llm_chain=chain), tools=tools, verbose=True, callback_manager=cb_manager
        )
        result = await agent_exec.arun(q)

        # Remove references to tool names that might be confused with a citation
        result = result.replace("[CognitiveSearch]", "").replace("[Employee]", "")

        return {"data_points": retrieve_results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log()}


class EmployeeInfoTool(CsvLookupTool):
    employee_name: str = ""

    def __init__(self, employee_name: str, callbacks: Callbacks = None):
        super().__init__(
            filename="data/employeeinfo.csv",
            key_field="name",
            name="Employee",
            description="useful for answering questions about the employee, their benefits and other personal information",
            callbacks=callbacks,
        )
        self.func = lambda _: "Not implemented"
        self.coroutine = self.employee_info
        self.employee_name = employee_name

    async def employee_info(self, name: str) -> str:
        return self.lookup(name)
