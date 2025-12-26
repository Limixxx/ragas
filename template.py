# 自定义类
from arguments import TestsetGenerationArguments
from doc.documents import load_data, data_transforms, get_knowledge_graph, get_persona, get_answers, get_queries
from model.GenerateModel import GenerateModel
from model.EmbeddingModel import EmbeddingModel

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas import RunConfig
from ragas.testset import TestsetGenerator

import os

def auto_testset_gen(args: TestsetGenerationArguments):
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

if __name__ == "__main__":
    testset_generation = {
        "docs": ["doc/中文文本相似性分析：Sentence-BERT应用指南.docx"],
        "test_size": 10,
        "output_file": "outputs/testset.json",
        "knowledge_graph": "outputs/knowledge_graph.json",
        "generator_llm": {
            "model_name": "deepseek-chat",
            "api_base": "https://api.deepseek.com",
            "api_key": "sk-4ac85c33c2c244328deb7a142b4b50f0",
        },
        "embeddings": {
            "model_name_or_path": ".venv/bge-small-zh-v1.5",
        },
        "language": "chinese",
    }
    from arguments import TestsetGenerationArguments
    args = TestsetGenerationArguments(**testset_generation)
    # 自动生成测试案例
    auto_testset_gen(args)