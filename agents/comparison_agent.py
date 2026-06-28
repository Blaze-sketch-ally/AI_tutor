from langsmith import traceable

from models.router import route


class ComparisonAgent:

    def __init__(
        self,
        provider="groq",
        model="llama-3.3-70b-versatile"
    ):

        self.provider = provider
        self.model = model

    @traceable(
        name="ComparisonAgent.generate_comparison"
    )
    def generate_comparison(
        self,
        topic: str,
        pdf_contexts: dict,
        history_text: str = "",
        summary: str = ""
    ):

        try:

            if not pdf_contexts:

                return (
                    "No PDF information available "
                    "for comparison."
                )

            pdf_text = ""

            for pdf_name, context in (
                pdf_contexts.items()
            ):

                pdf_text += f"""

==================================================
PDF NAME: {pdf_name}
==================================================

{context}

"""

            prompt = f"""
You are an expert academic comparison tutor.

Your task is to compare information
across MULTIPLE PDFs.

Conversation Summary:
{summary}

Chat History:
{history_text}

User Request:
{topic}

PDF Contents:

{pdf_text}

Instructions:

1. Identify ALL PDFs involved.

2. For each PDF:
   - Explain its viewpoint
   - Explain its explanation
   - Explain its contribution

3. Compare the PDFs.

4. Highlight:
   - Similarities
   - Differences
   - Unique Ideas

5. If one PDF contains information
   that another PDF does not,
   explicitly mention it.

6. If only one PDF contains relevant
   information, clearly state that.

7. Create a comparison table.

8. Be detailed and educational.

Output Format:

# PDFs Involved

# Overview

# PDF-wise Explanation

## PDF 1
...

## PDF 2
...

## PDF 3
...

# Similarities

# Differences

# Comparison Table

| Feature | PDF 1 | PDF 2 | PDF 3 |

# Key Insights

# Final Summary
"""

            response = route(
                prompt=prompt,
                provider=self.provider,
                model=self.model
            )

            return response

        except Exception as e:

            print(
                f"ComparisonAgent Error: {e}"
            )

            return (
                "Unable to generate comparison."
            )