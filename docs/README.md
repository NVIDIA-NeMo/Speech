# Documentation Process for NeMo

## Building the Documentation

1. Create and activate a virtual environment.

1. Install the documentation dependencies:

   ```console
   $ python3 -m pip install -r requirements/requirements_docs.txt
   ```

1. Build the documentation:

   ```console
   $ make -C docs html
   ```

## Checking for Broken Links

1. Build the documentation, as described in the preceding section, but use the following command:

   ```shell
   make -C docs clean linkcheck
   ```

1. Run the link-checking script:

   ```shell
   ./docs/check_for_broken_links.sh
   ```

If there are no broken links, then the script exits with `0`.

If the script produces any output, cut and paste the `uri` value into your browser to confirm
that the link is broken.

```json
{
  "filename": "nlp/text_normalization/nn_text_normalization.rst",
  "lineno": 247,
  "status": "broken",
  "code": 0,
  "uri": "https://research.fb.com/wp-content/uploads/2019/03/Neural-Models-of-Text-Normalization-for-Speech-Applications.pdf",
  "info": "400 Client Error: Bad Request for url: https://research.facebook.com/wp-content/uploads/2019/03/Neural-Models-of-Text-Normalization-for-Speech-Applications.pdf"
}
```

If the link is OK, and this is the case with many URLs that reference GitHub repository file headings,
then cut and paste the JSON output and add it to `docs/false_positives.json`.
Run the script again to confirm that the URL is no longer reported as a broken link.

There may be false positives due to Sphinx not being able to detect links from built html files.
Instead of adding those to the `docs/false_positives.json` file, it would be best to rewrite the
reference using a [:ref:](https://www.sphinx-doc.org/en/master/usage/referencing.html#role-ref).

For example, instead of writing `Modules <../api.html#modules>` to link to the modules section of
a `api.rst` file, write it as ``:ref:`Modules <asr-api-modules>` ``. And in the `api.rst` file, add
this label before the section being linked to:

```
.. _asr-api-modules:
```
