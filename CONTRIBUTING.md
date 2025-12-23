# Contributions are welcome!

We do all of NeMo's development in the open. Contributions from NeMo community are welcome.


# Pull Requests (PR) Guidelines

**Send your PRs to the `main` branch**

1) Make sure your PR does one thing. Have a clear answer to "What does this PR do?".
2) Read General Principles and style guide below
3) Make sure you sign your commits. E.g. use ``git commit -s`` when before your commit
4) Make sure all unittests finish successfully before sending PR ``pytest`` or (if your dev box does not have GPU) ``pytest --cpu`` from NeMo's root folder
5) Send your PR and request a review

## Unit tests
Quick tests (locally, while developing)
```
pytest
# If you don't have NVIDIA GPU do:
# pytest --cpu
```
Full tests, including pre-trained model downloads
```
pytest --with_downloads
```

## Whom should you ask for review:
1. For changes to NeMo's core: @ericharper, @titu1994, @blisc, or @okuchaiev  
1. For changes to NeMo's ASR collection: @titu1994, @redoctopus, @jbalam-nv, or @okuchaiev
1. For changes to NeMo's NLP collection: @MaximumEntropy, @ericharper, @ekmb, @yzhang123, @VahidooX, @vladgets, or @okuchaiev 
1. For changes to NeMo's TTS collection: @blisc, or @okuchaiev

Note that some people may self-assign to review your PR - in which case, please wait for them to add a review.

Your  pull requests must pass all checks and peer-review before they can be merged.

# General principles
1. **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
1. **Robust**: make it hard for users to make mistakes.
1. **Well-tested**: please add simple, fast unittests. Consider adding CI tests for end-to-end functionality.
1. **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to be reused.
1. **Readable**: code should be easier to read.
1. **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that NeMo supports. Give credit and link back to the code.
1. **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.

## Class naming conventions
* No “I”, “Interface”, “NM” nor “NeMo” pre/postfixes anywhere
* Core interfaces have simple names: Typing, Cloud, Serialization, FileIO*
* Core classes have the simplest names ever: NeuralModule, Model, Graph, Dataset, Loss, Module*
* Abstract classes in the Model hierarchy have Model postfix
* A config class for MyModel should be called MyModelConfig
* Leaf Neural Module classes have simple names without any postfixes (e.g. AudioPreprocess)
* Leaf Datasets have Dataset postfix (e.g. AudioToSpeechLabelDataset)
* Leaf Losses have Loss postfix (e.g. CTCLoss)
* Leaf Models do not have any postfix, just name (e.g. QuartzNet)

## Python style
We use ``black`` as our style guide. To check whether your code will pass style check (from the NeMo's repo folder) run:
``python setup.py style`` and if it does not pass run ``python setup.py style --fix``.

1. Include docstrings for every class and method exposed to the user.
1. Use Python 3 type hints for every class and method exposed to the user.
1. Avoid wild import: ``from X import *`` unless in ``X.py``, ``__all__`` is defined.
1. Minimize the use of ``**kwargs``.
1. ``RaiseError`` is preferred to ``assert``. Write: ```if X: raise Error``` instead of ```assert X```.
1. Classes are preferred to standalone methods.
1. Methods should be atomic. A method shouldn't be longer than 75 lines, e.g. can be fit into the computer screen without scrolling.
1. If a method has arguments that don't fit into one line, each argument should be in its own line for readability.
1. Add ``__init__.py`` for every folder.
1. F-strings are prefered to formatted strings.
1. Loggers are preferred to print. In NeMo, you can use logger from ``from nemo.utils import logging``
1. Private functions (functions start with ``_``) shouldn't be called outside its host file.
1. If a comment lasts multiple lines, use ``'''`` instead of ``#``.

# Collections
Collection is a logical grouping of related Neural Modules. It is a grouping of modules that share a domain area or semantics.
When contributing module to a collection, please make sure it belongs to that category. 
If you would like to start a new one and contribute back to the platform, you are very welcome to do so.  
