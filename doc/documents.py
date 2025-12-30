import asyncio
import os
import typing
import pandas as pd
from tqdm import tqdm

from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import BaseRagasLLM
from ragas.utils import num_tokens_from_string
from ragas.testset.graph import NodeType
from ragas.testset.transforms.engine import Parallel
from ragas.testset.transforms.extractors import EmbeddingExtractor, HeadlinesExtractor, SummaryExtractor, KeyphrasesExtractor
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.relationship_builders import CosineSimilarityBuilder, OverlapScoreBuilder
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.synthesizers.multi_hop import MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

from model.PromptTranslater import translate_prompts

def load_data(file_path):
    datas = []
    for path in tqdm(file_path):
        if path.endswith('.pdf'):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(path)
            datas.extend(loader.load())
        if path.endswith('.docx'):
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(path)
            datas.extend(loader.load())
    return datas

def data_transforms(
    llm: BaseRagasLLM,
    embedding: BaseRagasEmbedding,
    language: str,
):
    def filter_doc_with_num_tokens(node, min_num_tokens):
        return (
                node.type == NodeType.DOCUMENT and num_tokens_from_string(
            node.properties['page_content']) > min_num_tokens
        )

    # 从给定文本中提取最多 max_num 条关键标题，这些标题需能够将文本划分为若干独立章节。
    headline_extractor = HeadlinesExtractor(llm=llm, filter_nodes=lambda node: filter_doc_with_num_tokens(node, 500))
    # 按照headlines切分document为chunk，需要与HeadlinesExtractor的参数配合
    splitter = HeadlineSplitter(min_tokens=500, filter_nodes=lambda node: filter_doc_with_num_tokens(node, 500))
    # 总结document的内容
    summary_extractor = SummaryExtractor(llm=llm, filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100))
    # 向量化给定的文本
    # EmbeddingUsageEvent相关内容需要注释掉（有bug）
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding,
        property_name='summary_embedding',
        embed_property_name='summary',
        filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100),
    )
    # 从给定文本中提取最多 max_num 个关键短语。
    keyphrase_extractor = KeyphrasesExtractor(llm=llm, filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100))

    # 从给定文本中提炼核心主题与核心概念。
    theme_extractor = ThemesExtractor(llm=llm, filter_nodes=lambda node: node.type == NodeType.CHUNK)
    # 从给定文本中提取命名实体，输出结果仅限排名靠前的实体，且实体数量不得超过指定上限。
    ner_extractor = NERExtractor(llm=llm, filter_nodes=lambda node: node.type == NodeType.CHUNK)

    # 给定一份文档摘要和节点内容，请在 1 至 5 的范围内为该节点内容与摘要相关性评分。
    node_filter = CustomNodeFilter(llm=llm, filter_nodes=lambda node: node.type == NodeType.CHUNK)

    # 基于文本嵌入向量（Embedding）的余弦相似度，为知识图谱（KnowledgeGraph）中符合条件的节点建立双向相似性关系
    cosine_sim_builder = CosineSimilarityBuilder(
        property_name='summary_embedding',
        new_property_name='summary_similarity',
        threshold=0.7,
        filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100),
    )
    # 基于知识图谱节点的指定属性（默认是实体列表entities），通过字符串相似度计算筛选有效重叠项，最终为符合阈值的节点对生成带重叠分数和重叠项的单向相似性关系
    ner_overlap_sim = OverlapScoreBuilder(threshold=0.01, filter_nodes=lambda node: node.type == NodeType.CHUNK)

    asyncio.run(
        translate_prompts(
            prompts=[headline_extractor, summary_extractor, theme_extractor, ner_extractor, node_filter],
            target_lang=language,
            llm=llm,
            adapt_instruction=True,
        )
    )

    return [
        headline_extractor,
        splitter,
        summary_extractor,
        keyphrase_extractor,
        node_filter,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]

def get_knowledge_graph(documents, transforms, local_file, run_config):
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms

    if os.path.exists(local_file):
        print(f'Loading knowledge graph from {local_file}')
        return KnowledgeGraph.load(local_file)
    # convert the documents to Ragas nodes
    nodes = []
    for doc in documents:
        node = Node(
            type=NodeType.DOCUMENT,
            properties={
                'page_content': doc.page_content,
                'document_metadata': doc.metadata,
            },
        )
        nodes.append(node)

    kg = KnowledgeGraph(nodes=nodes)

    # apply transforms and update the knowledge graph
    apply_transforms(kg, transforms, run_config=run_config)

    # save the knowledge graph
    output_path = os.path.dirname(local_file)
    os.makedirs(output_path, exist_ok=True)
    kg.save(local_file)
    print(f'Knowledge graph saved to {local_file}')
    return kg

"""
    根据所提供的摘要，生成一个有可能接触到该内容或从中获益的目标用户画像。画像需包含一个独特的姓名，以及一段关于其身份的简明角色描述。
"""
def get_persona(llm, kg, language):
    from ragas.prompt import PydanticPrompt, StringIO
    from ragas.testset.persona import Persona
    from ragas.testset.persona import PersonaGenerationPrompt, generate_personas_from_kg

    class PersonaGenerationPromptZH(PydanticPrompt[StringIO, Persona]):
        instruction: str = (
            '使用提供的摘要，生成一个可能会与内容互动或从中受益的角色。包括一个独特的名字和一个简洁的角色描述。')
        input_model: typing.Type[StringIO] = StringIO
        output_model: typing.Type[Persona] = Persona
        examples: typing.List[typing.Tuple[StringIO, Persona]] = [(
            StringIO(text='《数字营销指南》解释了在各种在线平台上吸引受众的策略。'),
            Persona(
                name='数字营销专家',
                role_description='专注于吸引受众并在线上提升品牌。',
            ),
        )]

    if language == 'chinese':
        persona_prompt = PersonaGenerationPromptZH()
    else:
        persona_prompt = PersonaGenerationPrompt()

    return generate_personas_from_kg(llm=llm, kg=kg, num_personas=3, persona_generation_prompt=persona_prompt)

def get_queries(llm: BaseRagasLLM, kg: KnowledgeGraph, language: str):

    # 给定一份主题列表，以及一份附带角色说明的用户画像列表，请根据用户画像的角色描述，将每一个用户画像与对应的相关主题进行匹配。
    single_hop = SingleHopSpecificQuerySynthesizer(llm=llm)
    """
    从至少两个不同的列表中选取概念进行配对，形成组合。操作说明
    - 查阅每个节点下的所有概念。
    - 找出在逻辑上可以关联或形成对比的概念。
    - 组合需包含来自不同节点的概念。
    - 每个组合必须涵盖至少两个及以上节点中的各一个概念。
    - 清晰、简洁地列出所有组合。
    - 同一组合不得重复列出。
    最后给定一份主题列表，以及一份附带角色说明的用户画像列表，请依据用户画像的角色描述，将每个用户画像与对应的相关主题进行关联匹配。
    """
    multi_hop_abs = MultiHopAbstractQuerySynthesizer(llm=llm)
    """
    给定一份主题列表与一份附带角色说明的用户画像列表，请依据各用户画像的角色描述，将其与对应的相关主题进行关联匹配。
    根据指定条件（用户画像、主题、风格、篇幅）以及提供的上下文，生成一个多跳问答内容。其中，主题是从上下文里提取或生成的一组短语，用于体现所选上下文适用于构建多跳问题。请确保问题中明确融入这些主题。
    操作说明
    - 生成多跳问题：利用提供的上下文片段与主题，构建一个需要整合多个片段（如<1跳>和<2跳>）信息才能解答的问题。确保问题明确包含一个或多个主题，并体现出主题与上下文的相关性。
    - 生成答案：仅基于所提供的上下文内容，撰写一份详实且忠实于原文的问题答案。禁止添加任何未在原文中直接体现或无法从原文推导得出的信息。
    - 多跳上下文标签规则
    - 每个上下文片段均标注为<1跳>、<2跳>等类别。
    - 确保所生成的问题需调用至少两个片段的信息，并将这些信息进行有意义的关联。
    - 补充上下文（若有提供）：若提供了大语言模型上下文（llm_context），则需以此为指导，确定问题的生成类型（例如比较类问题、因果类问题、应用类问题），并据此规划答案结构。请注意，答案内容仍需完全取自所提供的上下文。
    """
    multi_hop_spec = MultiHopSpecificQuerySynthesizer(llm=llm)

    asyncio.run(
        translate_prompts(
            prompts=[
                single_hop,
                multi_hop_abs,
                multi_hop_spec,
            ],
            target_lang=language,
            llm=llm,
            adapt_instruction=True,
        )
    )

    default_queries = [
        single_hop,
        multi_hop_abs,
        multi_hop_spec,
    ]
    if kg is not None:
        available_queries = []
        for query in default_queries:
            if query.get_node_clusters(kg):
                available_queries.append(query)
    else:
        available_queries = default_queries

    return [(query, 1 / len(available_queries)) for query in available_queries]

def get_answers(testset_df, generator_llm, language: str):
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Answer in {language}.
    Question: {question}
    Context: {contexts}
    Answer:
    """

    items = []
    for i in tqdm(range(len(testset_df)), desc='Generating Answers'):
        row = testset_df.iloc[i]
        question = row['user_input']
        contexts = '\n'.join(row['reference_contexts'])

        # Combine question and contexts as input for the LLM
        input_text = template.format(language=language, question=question, contexts=contexts)

        # Generate the answer using the generator LLM
        answer = generator_llm.invoke(input_text)
        items.append({
            'user_input': question,
            'retrieved_contexts': row['reference_contexts'],
            'response': answer.content,
            'reference': row['reference'],
        })

    return pd.DataFrame.from_dict(items)