"""Agent de recherche web (LangChain + DeepSeek).

Un agent « chercheur » qui, à partir d'une question, cherche des documents sur
le web puis les lit pour rassembler l'information pertinente.
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

from app.agent.tools import read_url, search_web
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL

SYSTEM_PROMPT = """Tu es un chercheur technique.
Tu réponds à une question en te basant UNIQUEMENT sur des documents trouvés sur le web.

Méthode :
1. Utilise l'outil `web_search` pour trouver des documents (PDF en priorité).
2. Utilise `read_document` pour lire le document le plus pertinent.
3. Si l'information manque, cherche et lis un autre document.

Rédige ensuite une synthèse claire et factuelle, en citant les sources (URL)."""


def build_llm() -> ChatDeepSeek:
    """Construit le LLM DeepSeek (nécessite une clé API)."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY manquante : renseigne-la dans .env ou l'environnement."
        )
    return ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, temperature=0)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


@tool
def web_search(query: str) -> str:
    """Cherche des documents (PDF en priorité) sur le web et renvoie les résultats."""
    results = search_web(query)
    if not results:
        return "Aucun résultat."
    return "\n".join(
        f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}"
        for i, r in enumerate(results, start=1)
    )


@tool
def read_document(url: str) -> str:
    """Télécharge un document (PDF ou page web) et renvoie son texte."""
    return read_url(url)["text"]


def build_agent() -> AgentExecutor:
    """Assemble l'agent chercheur (LLM + outils de recherche et de lecture)."""
    tools = [web_search, read_document]
    agent = create_tool_calling_agent(build_llm(), tools, build_prompt())
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=8,
        handle_parsing_errors=True,
    )


def research(question: str) -> str:
    """Lance l'agent chercheur sur `question` et renvoie sa synthèse."""
    result = build_agent().invoke({"input": question})
    return result["output"]
