from arguments import TestsetGenerationArguments, EvaluationArguments
from model.GenerateModel import GenerateModel
from model.EmbeddingModel import EmbeddingModel

import os

from ragas import RunConfig
from ragas.llms import LangchainLLMWrapper

def auto_testset_gen(args: TestsetGenerationArguments):
    from doc.documents import load_data, data_transforms, get_knowledge_graph, get_persona, get_answers, get_queries

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.testset import TestsetGenerator

    docs = load_data(args.docs)
    generator = GenerateModel.load(**args.generator_llm)
    embedding = EmbeddingModel.load(**args.embeddings)

    wrapped_generator = LangchainLLMWrapper(generator)
    wrapped_embedding = LangchainEmbeddingsWrapper(embedding)

    transforms = data_transforms(
        wrapped_generator,
        wrapped_embedding,
        args.language,
    )

    run_config = RunConfig(timeout=600, max_retries=10, max_wait=120, max_workers=1, log_tenacity=True)
    knowledge_graph = get_knowledge_graph(docs, transforms, args.knowledge_graph, run_config)
    persona_list = get_persona(llm=wrapped_generator, kg=knowledge_graph, language=args.language)

    queries = get_queries(wrapped_generator, knowledge_graph, args.language)
    testset_generator = TestsetGenerator(
        llm=wrapped_generator, embedding_model=wrapped_embedding, knowledge_graph=knowledge_graph,
        persona_list=persona_list
    )
    testset = testset_generator.generate(
        testset_size=args.test_size,
        query_distribution=queries,
        run_config=run_config,
        with_debugging_logs=True,
        raise_exceptions=True,
    )

    testset_df = testset.to_pandas()
    output_path = os.path.dirname(args.output_file)
    os.makedirs(output_path, exist_ok=True)
    testset_df.to_json(args.output_file, indent=4, index=False, orient='records', force_ascii=False)

    # get answer
    testset_with_answer = get_answers(testset_df, generator, args.language)
    testset_with_answer.to_json(
        args.output_file.replace('.json', '_with_answer.json'),
        indent=4,
        index=False,
        orient='records',
        force_ascii=False,
    )
    print("auto_testset_gen OK")

def auto_eval(args: EvaluationArguments):
    from model.PromptTranslater import translate_prompts
    from utils.utils import dynamic_import

    from ragas import evaluate

    from datasets import Dataset
    import asyncio

    dataset = Dataset.from_json(args.testset_file)
    critic = GenerateModel.load(**args.critic_llm)
    embedding = EmbeddingModel.load(**args.embeddings)
    metrics = dynamic_import('ragas.metrics', args.metrics)

    asyncio.run(
        translate_prompts(
            prompts=metrics,
            target_lang=args.language,
            llm=LangchainLLMWrapper(critic),
            adapt_instruction=True,
        )
    )
    run_config = RunConfig(timeout=600, max_retries=2, max_wait=60, max_workers=1)
    score = evaluate(
        dataset,
        metrics=metrics,
        llm=critic,
        embeddings=embedding,
        run_config=run_config,
    )
    score_df = score.to_pandas()
    print(score_df)

    output_path = args.testset_file.replace('.json', '_score.json')
    score_df.to_json(output_path, indent=4, index=False, orient='records', force_ascii=False)

    print(f'Eval score saved to {output_path}')



if __name__ == "__main__":
    """
    testset_generation = {
        #"docs": ["doc/中文文本相似性分析：Sentence-BERT应用指南.docx"],
        #"language": "chinese",
        "docs": ["doc/Fine-Tuning_distilBERT_for_Enhanced_Sentiment_Clas.pdf"],
        "language": "english",
        "test_size": 10,
        "output_file": "outputs/testset.json",
        "knowledge_graph": "outputs/knowledge_graph.json",
        "generator_llm": {
            #"api_base": "https://api.deepseek.com",
            #"api_key": "sk-53840dc61fc7483697e92c489c35d117",
            #"model_name": "deepseek-chat",
            "api_base": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key": "4f9ca511-527b-4b13-a068-2bd4834f85ad",
            "model_name": "doubao-seed-1-6-251015",
        },
        "embeddings": {
            "model_name_or_path": ".venv/bge-small-zh-v1.5",
        },
    }
    from arguments import TestsetGenerationArguments
    args = TestsetGenerationArguments(**testset_generation)
    # 自动生成测试案例
    auto_testset_gen(args)
    """

    evaluation = {
        "testset_file": "outputs/testset_with_answer.json",
        "critic_llm": {
            "api_base": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key": "4f9ca511-527b-4b13-a068-2bd4834f85ad",
            "model_name": "doubao-seed-1-6-251015",
        },
        "embeddings": {
            "model_name_or_path": ".venv/bge-small-zh-v1.5",
        },
        "metrics": [
            "Faithfulness",
            "AnswerRelevancy",
            "ContextPrecision",
            "AnswerCorrectness",
        ],
        "language": "english",
    }
    from arguments import EvaluationArguments
    args = EvaluationArguments(**evaluation)
    # 自动评测
    auto_eval(args)