from dataclasses import dataclass, field
from typing import List


@dataclass
class RagContext:
    user_input: str = ""
    reference: str = ""
    retrieved_contexts: List[str] = field(default_factory=list)

"""
Context Precision
上下文精确率是一种评估检索器在检索到的上下文中，针对给定查询将相关片段排在不相关片段之前的能力的指标。具体来说，它评估的是检索到的上下文中的相关片段在排名中处于顶部的程度。
"""

async def context_precision(llm, ctx: RagContext):
    from ragas.metrics.collections import ContextPrecision

    # Create metric
    scorer = ContextPrecision(llm=llm)

    # Evaluate
    result = await scorer.ascore(
        user_input=ctx.user_input,
        reference=ctx.reference,
        retrieved_contexts=ctx.retrieved_contexts
    )
    print(f"Context Precision Score: {result.value}")
    return result

from model.GenerateModel import InstructorModel

if __name__ == "__main__":
    llm_config = {
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key": "4f9ca511-527b-4b13-a068-2bd4834f85ad",
        "model_name": "doubao-seed-1-6-251015",
    }
    llm = InstructorModel.load(**llm_config)

    context = RagContext()
    context.user_input = "Where is the Eiffel Tower located?"
    context.reference = "The Eiffel Tower is located in Paris."
    context.retrieved_contexts = [
        "The Eiffel Tower is located in Berlin.",
        "The Brandenburg Gate is located in Berlin."
    ]

    import asyncio

    # Context Precision
    asyncio.run(context_precision(llm, context))
