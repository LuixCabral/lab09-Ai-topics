import os
import google.generativeai as genai


def hyde_document(query: str) -> str:
    """Generate a hypothetical technical document (HyDE) using Gemini.

    If GEMINI_API_KEY is configured, it tries a real LLM call.
    Otherwise it falls back to a simple template.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "Escreva um paragrafo tecnico de manual medico que responda: "
                f"{query}. Use jargao clinico e termos formais."
            )
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                ),
            )
            text = response.text.strip()
            if text:
                return text
        except Exception:
            pass

    return (
        "Manual tecnico: quadro compativel com cefaleia primaria, "
        "associada a fotofobia e dor pulsante. Avaliar sinais de alarme, "
        "considerar enxaqueca e tratamento sintomatico."
    )
