import re
from functools import cache

TEMPLATE_VAR_VALIDATION_PAT = re.compile(r'^\{_[A-Za-z][A-Za-z0-9_]*_\}$')
TEMPLATE_VAR_SEARCH_PAT = re.compile('({_[^}]+_})')


class ChatTemplateMixin:
    def apply_chat_template(self, messages):
        assert self.chat_template is not None
        return tokenize_with_chat_template(self, messages, self.chat_template)

    @property
    def has_chat_template(self):
        return self.chat_template is not None


@cache
def is_template_var(s):
    # It should start with {_ and end with _}, be non-empty and not contain { or } within.
    return re.match(TEMPLATE_VAR_VALIDATION_PAT, s)


def extract_template_parts(template, skip_empty=True):
    for part in re.split(TEMPLATE_VAR_SEARCH_PAT, template):
        # skip empty parts
        if skip_empty and part == '':
            continue
        yield part


def strip_template_wrap(s):
    if not is_template_var(s):
        return s
    # Strip the "{_" prefix and the "_}" suffix
    return s[2:-2]


def render_chat_turn(message, template):
    """Renders a chat turn based on template

    Args:
        message (Dict)
        e.g. {'role': ['user'], 'content': ['What is your favourite fruit?']},
        template (Str):
            "[INST] {_content_} [/INST]",

    Returns:
        (str, token_id/None): the template formatted message
        e.g.
            "[INST] What is your favourite fruit? [/INST]", None
    """
    ans = []
    for i, template_part in enumerate(extract_template_parts(template)):
        if is_template_var(template_part):
            template_part = strip_template_wrap(template_part)
            if template_part == 'content':
                ans.append(message['content'])
            else:
                # assert i == len(template_parts) - 1, "unsupported"
                yield ''.join(ans), template_part
                ans = []
        else:
            # Otherwise it is literal string
            ans.append(template_part)
    yield ''.join(ans), None


def encode_string_with_special_token(tokenizer, inputs, special_token):
    """
    Tokenizes a string or a list of string into their corresponding token_ids
    and appends (at the end) a special_token if present.

    Args:
        tokenizer: (SPM)
        inputs: (Str, List[Str])
        e.g. "Alex" or ["Alex", "nvidia"]
        special_token: (Str):
        e.g. "eos"

        Returns:
         (list[int]): list of token_ids
         e.g.
            input="Alex", special_token="eos"
            Alex->[3413]
            eos->[2]

            Will return the following:
            [3413, 2]
    """
    ans = []
    if isinstance(inputs, str) and inputs != '':
        ans += tokenizer.text_to_ids(inputs)
    elif isinstance(inputs, list) and len(inputs) > 0:
        ans += tokenizer.text_to_ids(''.join(inputs))
    if special_token is not None:
        # TODO(@akoumparouli): limit which attributes user-defined string can query.
        assert hasattr(tokenizer, special_token), f"Special_token {special_token} is not part of tokenizer"
        ans += [getattr(tokenizer, special_token)]
    return ans


def tokenize_with_chat_template(tokenizer, messages, template):
    assert is_chat_input(messages), "Expected input to be chat-template"
    assert len(messages) > 0, "Expected non-empty messages"
    assert 'roles' in template, "Expected template to have key `roles`."
    ans = []
    encode = lambda x, y: encode_string_with_special_token(tokenizer, x, y)
    if 'prefix' in template:
        for part, special_token in render_chat_turn('', template['prefix']):
            ans += encode(part, special_token)
    buffer = []
    for message in messages:
        assert message['role'] in template['roles'], (message['role'], template['roles'])
        msg_template = template['roles'][message['role']]
        for templated_messages, special_token in render_chat_turn(message, msg_template):
            buffer += [templated_messages]
            if special_token is not None:
                ans += encode(buffer, special_token)
                buffer = []
    # handle tail
    ans += encode(buffer, None)
    assert len(ans) > 0, 'Expected non-empty output'
    return ans


def extract_turns(messages, axis):
    """
    a collated messages can have multiple chat messages in each dict,
    this extracts (vertically) one of them, for example:

    messages = [
        {'role': ['user', 'user'], 'content': ['What is your favourite condiment?', 'What is your favourite fruit?']},
        {'role': ['assistant', 'assistant'], 'content': ["Well, I'm quite partial to a ", "good squeeze of fresh lemon"]},
        {'role': ['user', 'user'], 'content': ['Do you have mayonnaise recipes?', 'Do you have tomato salad recipes?']}
    ]
    ans = extract_turns(messages, axis=1)

    ans = [
        {'role': ['user'], 'content': ['What is your favourite fruit?']},
        {'role': ['assistant'], 'content': ["good squeeze of fresh lemon"]},
        {'role': ['user'], 'content': ['Do you have tomato salad recipes?']}
    ]
    """
    ans = []
    for turn in messages:
        ans.append({k: v[axis] for k, v in turn.items()})
    return ans


def explode_chat_template_input(messages):
    """
    Example input
    [
       {'role': ['user', 'user'], 'content': ['What is your favourite condiment?', 'What is your favourite fruit?']},
       {'role': ['assistant', 'assistant'], 'content': ["Well, I'm quite partial to a ", "good squeeze of fresh lemon"]},
       {'role': ['user', 'user'], 'content': ['Do you have mayonnaise recipes?', 'Do you have tomato salad recipes?']}
    ]

    Notice the 2D axis system of the messages variable, one for the list and one for each item in the list (i.e.
    the 'content' contains multiple messages).
    """
    assert isinstance(messages, list), "Expected messages to be a list"
    assert len(messages) > 0, "Expected non empty messages"
    assert all(map(lambda x: isinstance(x, dict), messages)), "Expected messages to contain dicts"
    assert all(
        map(lambda x: 'role' in x and 'content' in x, messages)
    ), "Expected messages each dict to contain 'role' and 'content' fields"
    n = len(messages[0]['role'])
    assert all(
        map(lambda x: len(x['role']) == n, messages)
    ), "Expected all batch messages to contain equal number of roles in all turns"
    for i in range(n):
        yield extract_turns(messages, axis=i)


def is_chat_input(messages):
    # TOOD(@akoumparouli): improve validation.
    return isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict)
