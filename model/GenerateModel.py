from langchain_openai import ChatOpenAI, OpenAI


class GenerateModel:

    @staticmethod
    def load(**kw):
        api_base = kw.get('api_base', None)
        if api_base:
            return ChatOpenAI(
                model=kw.get('model_name', ''),
                base_url=api_base,
                api_key=kw.get('api_key', 'EMPTY'),
            )
        return None
        # TODO 后续支持本地模型

class InstructorModel:

    @staticmethod
    def load(**kw):
        model_name = kw.get('model_name', '')
        api_base = kw.get('api_base', None)
        api_key = kw.get('api_key', None)

        chat = OpenAI(
            model=model_name,
            base_url=api_base,
            api_key=api_key,
        )

        from ragas.llms import llm_factory
        return llm_factory(model_name, client=chat)