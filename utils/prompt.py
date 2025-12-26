
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download necessary NLTK data resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

"""
将文本中最多 n 个随机单词替换为来自WordNet的同义词。
"""
def synonym_replacement(text, n=1):
    """
    Replaces up to n random words in text with their synonyms from WordNet.
    """
    words = word_tokenize(text)
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

"""
使用 WordNet 获取并清洗给定单词的同义词
"""
def get_synonyms(word):
    """
    Retrieves and cleans synonyms for a given word using WordNet.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char.isalnum() or char == ' '])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

"""
随机打乱输入文本中的单词顺序
"""
def shuffle_words(text):
    """
    Randomly shuffles the order of words in the input text.
    """
    words = word_tokenize(text)
    random.shuffle(words)
    return ' '.join(words)

"""
以概率 p 从文本中随机删除单词
"""
def delete_random_words(text, p=0.1):
    """
    Randomly deletes words from text with probability p.
    """
    words = word_tokenize(text)
    words = [w for w in words if random.random() > p]
    return ' '.join(words)

"""
在文本的随机位置插入 n 个常用词
"""
def insert_random_words(text, n=2):
    """
    Inserts n common words at random positions in the text.
    """
    common_words = ['also', 'very', 'then', 'moreover', 'however', 'thus', 'indeed']
    words = word_tokenize(text)
    for _ in range(n):
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, random.choice(common_words))
    return ' '.join(words)

"""
交换文本中 n 个随机的单词对
"""
def random_swap(text, n=1):
    """
    Swaps n random pairs of words in the text.
    """
    words = word_tokenize(text)
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)


# Mapping of augmentation strategy IDs to functions
strategies = {
    0: synonym_replacement,
    1: shuffle_words,
    2: delete_random_words,
    3: insert_random_words,
    4: random_swap
}

"""
在保持输入内容不变的前提下，对指令执行数据增强操作
"""
def augment_text(text, strategy_func=None):
    """
    Applies augmentation to an text while keeping input unchanged.
    """
    if strategy_func is None:
        return text

    try:
        augmented_text = strategy_func(text)
    except Exception as e:
        logger.error(f"Error during augmentation: {e}, using original instruction")
        augmented_text = text

    return augmented_text

# Predefined prompts for QA instruction replacement
qa_prompts = [
    "This constitutes a context-dependent question-answer assignment; kindly ensure responses correlate strictly with incoming queries.",
    "The current task involves contextual query processing - please align all answers directly with provided questions.",
    "Kindly treat this as scenario-specific QA operation requiring strict adherence to input inquiries.",
    "Operate within given context boundaries for answering, maintaining precise correspondence with posed questions.",
    "Execute response generation based on contextual framework while preserving strict alignment with received questions.",
    "Perform context-aware answer formulation that mirrors intellectual contours of submitted inquiries.",
    "Conduct situation-bound information retrieval ensuring output corresponds proportionally to input prompts.",
    "Apply contextual analysis methodology where solutions derive exclusively from parameters established by questioning inputs.",
    "Maintain situational fidelity in responses through rigorous cross-referencing against originating queries.",
    "Implement context-sensitive answering protocol prioritizing direct correspondence with supplied interrogatives."
]

def replace_instruction(items, target_prompts=qa_prompts, exclude_prompts=[]):
    """
    Replaces instructions for context-based QA items with unique prompts per bucket.
    """
    available = [p for p in target_prompts if p not in exclude_prompts]
    if len(available) < 2:
        available = target_prompts  # Reset if insufficient new prompts
    prompt_pair = random.sample(available, 2)
    exclude_prompts.update(prompt_pair)

    replaced_items = []
    for item in items:
        chosen_prompt = random.choice(prompt_pair)
        replaced_items.append({
            "instruction": chosen_prompt,
            "input": item["input"],
            "output": item["output"]
        })
    return replaced_items