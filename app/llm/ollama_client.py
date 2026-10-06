"""Client Ollama (LLM local) et prompt de génération."""

from functools import lru_cache

import ollama

from app.config import OLLAMA_HOST, OLLAMA_MODEL

# Phrase que le modèle doit renvoyer quand l'information est absente du contexte.
ABSENCE_PHRASE = "Cette information n'est pas présente dans la documentation fournie."

PROMPT_TEMPLATE = f"""Tu es un assistant technique. Réponds STRICTEMENT à partir du contexte fourni ci-dessous.
N'ajoute AUCUNE information, exemple ou détail qui n'est pas explicitement présent dans ce contexte, même si cela te semble plausible ou utile.
Si le contexte ne contient pas la réponse complète à la question, dis clairement : "{ABSENCE_PHRASE}"

Contexte:
{{context}}

Question: {{question}}

Réponse (strictement basée sur le contexte ci-dessus, sans ajout):"""

CONTEXT_SEPARATOR = "\n\n---\n\n"


@lru_cache(maxsize=1)
def get_client() -> ollama.Client:
    """Client Ollama (instancié une seule fois, à la demande)."""
    return ollama.Client(host=OLLAMA_HOST)


def generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Envoie un prompt au LLM et renvoie sa réponse texte."""
    response = get_client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def generate_answer(question: str, context_chunks: list[str], model: str = OLLAMA_MODEL) -> str:
    """Construit le contexte, applique le prompt strict et interroge le LLM."""
    context = CONTEXT_SEPARATOR.join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return generate(prompt, model=model)
