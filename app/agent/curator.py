"""Curateur : reformule la recherche en une fiche claire et concise (DeepSeek),
puis s'auto-évalue.
"""

from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agent.researcher import build_llm

REWRITE_SYSTEM = """Tu es un rédacteur technique.
À partir d'un contenu brut de recherche, rédige une fiche en Markdown, courte et claire.

Règles :
- Ne garde que les informations importantes et factuelles.
- Sois concis : pas de remplissage ni d'introduction inutile.
- Conserve les chiffres, noms et unités exacts.
- Termine par une section « Sources » listant les URL utilisées (si fournies)."""

REVIEW_SYSTEM = """Tu es un relecteur technique.
Évalue si la fiche répond fidèlement et clairement à la question.

Réponds STRICTEMENT sur deux lignes :
VERDICT: OK ou KO
RAISON: <une phrase courte>"""


@dataclass
class CurationResult:
    """Résultat de la curation : la fiche et le verdict du relecteur."""

    content: str
    approved: bool
    reason: str


def build_rewrite_chain(llm=None):
    """Chaîne LLM qui réécrit un contenu brut en fiche concise."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_SYSTEM),
            ("human", "Question : {question}\n\nContenu brut :\n{raw}"),
        ]
    )
    return prompt | (llm or build_llm()) | StrOutputParser()


def build_review_chain(llm=None):
    """Chaîne LLM qui évalue la fiche produite."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REVIEW_SYSTEM),
            ("human", "Question : {question}\n\nFiche :\n{content}"),
        ]
    )
    return prompt | (llm or build_llm()) | StrOutputParser()


def rewrite(question: str, raw_material: str, llm=None) -> str:
    """Réécrit le contenu brut en une fiche concise."""
    return (
        build_rewrite_chain(llm)
        .invoke({"question": question, "raw": raw_material})
        .strip()
    )


def parse_review(text: str) -> tuple[bool, str]:
    """Extrait le verdict et la raison d'une réponse de relecture."""
    approved = False
    reason = ""
    for line in text.splitlines():
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            approved = line.split(":", 1)[1].strip().upper().startswith("OK")
        elif upper.startswith("RAISON:"):
            reason = line.split(":", 1)[1].strip()
    return approved, reason


def review(question: str, content: str, llm=None) -> tuple[bool, str]:
    """Fait relire la fiche par le LLM et renvoie `(approuvée, raison)`."""
    text = build_review_chain(llm).invoke({"question": question, "content": content})
    return parse_review(text)


def curate(question: str, raw_material: str, llm=None) -> CurationResult:
    """Réécrit la recherche puis vérifie que la fiche est fidèle et exploitable."""
    content = rewrite(question, raw_material, llm=llm)
    approved, reason = review(question, content, llm=llm)
    return CurationResult(content=content, approved=approved, reason=reason)
