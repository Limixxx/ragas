from langchain_openai import ChatOpenAI

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