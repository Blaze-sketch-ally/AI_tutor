from models.groq_client import generate as groq_generate
from models.hf_client import generate as hf_generate


def route(
    prompt: str,
    provider: str = "groq",
    model: str | None = None
):
    """
    Routes requests to the selected model provider.
    """

    provider = provider.lower()

    if provider == "groq":

        if model is None:
            model = "llama-3.3-70b-versatile"

        return groq_generate(
            prompt=prompt,
            model=model
        )

    elif provider == "hf":

        if model is None:
            model = "google/gemma-2-9b-it"

        return hf_generate(
            prompt=prompt,
            model=model
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider}"
        )